from .MCTS import MCTS, Node
from .NeuralNet import NeuralNet
from .State import State

from ...env import env_2048
from ...utils.progress_bar import ProgressBar

import numpy as np
import matplotlib.pyplot as plt

def train_deep_mcts(
    num_episodes: int,
    num_warm_up: int,
    num_mcts_simulations: int,
    neural_net: NeuralNet,
    saving_path: str
) -> None:

    env = env_2048()

    training_losses = []
    max_tiles = []
    episode_lengths = []

    for episode in range(num_episodes + num_warm_up):
        print(f'\nEpisode {episode + 1} {"(warm up)" if episode < num_warm_up else ""}:\n')
        mcts = MCTS(neural_net, 1 if episode < num_warm_up else num_mcts_simulations, warm_up=episode < num_warm_up)
        states, policy_targets, value_targets = [], [], []
        env.reset()
        state = State(env.grid)
        current_node = Node(state)
        move_count = 0
        pbar = ProgressBar(4000)

        while not current_node.state.is_terminal():
            if current_node.state.player == 1:
                states.append(current_node.state.clone())
                
                max_tile_seen = mcts.run(current_node)

                if sum([child.visit_count for child in current_node.children]) != 0:
                    policy_target = np.zeros(4)
                    for move in range(4):
                        childs = [child.visit_count for child in current_node.children if child.state.last_move == move]
                        policy_target[move] = childs[0] if len(childs) > 0 else 0
                    
                    policy_target /= policy_target.sum()

                else:
                    policy_target = np.ones(4) / 4

                policy_targets.append(policy_target)
                
                max_p = 0.0
                moves = [child.state.last_move for child in current_node.children]
                for i, p in enumerate(policy_target):
                    if p >= max_p and i in moves:
                        best_move = i 
                        max_p = p
              
                current_node = [child for child in current_node.children if child.state.last_move == best_move][0]
                value_targets.append(current_node.value)
                move_count += 1
                pbar.update({'max tile seen': max_tile_seen}, move_count)

            else:
                state = current_node.state.play_move()
                child_node = Node(state, parent=current_node, policy=1.0)
                current_node.children.append(child_node)
                current_node = child_node
        
        avg_loss = neural_net.train(states, policy_targets, value_targets)
        training_losses.append(avg_loss)
        episode_lengths.append(move_count)
        max_tiles.append(state.max_reached())

        print(f'\n--- max tile reached: {state.max_reached()} --- move count: {move_count} --- average loss: {avg_loss}\n\n#########\n')
        neural_net.save(f"{saving_path}MCTS_CNN_2048.pt")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(num_episodes), training_losses, label="Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(range(num_episodes), max_tiles, label="Max Tile")
    plt.xlabel("Episode")
    plt.ylabel("Max Tile")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(num_episodes), episode_lengths, label="Episode lengths")
    plt.xlabel("Episode")
    plt.ylabel("length")
    plt.legend()

    plt.tight_layout()
    plt.show()