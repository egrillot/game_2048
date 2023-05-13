from .MCTS import MCTS, Node
from .State import State
from ...env import env_2048
from ...utils import ProgressBar
from ...utils import build_training_overview
from ..mcts import Naive_MCTS
from ...utils import vectorize_action, vectorize_w
from .MCTSAgent import MCTSAgent

import numpy as np
import matplotlib.pyplot as plt

def warm_up(
    num_episodes: int,
    neural_net: MCTSAgent,
    saving_path: str,
    naive_mcts_num_sample=20,
    naive_mcts_depth=200
) -> None:
    print(f'Warm up, naive mcts sample={naive_mcts_num_sample}, naive mcts depth={naive_mcts_depth}\n')

    for i in range(num_episodes):
        print(f'Game {i+1}/{num_episodes}:\n')
        naive_mcts = Naive_MCTS(num_sample=naive_mcts_num_sample, depth=naive_mcts_depth)

        grids, actions, successfull_game = naive_mcts.play_game()
        nb_move = len(actions)
        print('\nTrainig...')
        avg_loss = neural_net.train(
            grids, 
            [vectorize_action(action) for action in actions], 
            vectorize_w(successfull_game, nb_move),
            epochs=100
        )
        print(f"Average loss during the training after 100 epochs: {avg_loss}")
        neural_net.save(f"{saving_path}warmed_up_model.pt")

def train_deep_mcts(
    num_episodes: int,
    num_mcts_simulations: int,
    model: MCTSAgent,
    saving_path: str
) -> None:

    env = env_2048()

    training_losses = []
    max_tiles = []
    episode_lengths = []
    overview = dict()

    for episode in range(num_episodes):
        print(f'\nEpisode {episode + 1}:\n')
        mcts = MCTS(model, num_mcts_simulations)
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
                pbar.update({'max tile seen': max_tile_seen, 'tree max depth': mcts.get_depth()}, move_count)

            else:
                state, _ = current_node.state.play_move()
                child_node = Node(state, parent=current_node, policy=1.0)
                current_node.children.append(child_node)
                current_node = child_node
        
        max_tile = state.max_reached()
        avg_loss = model.train(states, policy_targets, value_targets)
        training_losses.append(avg_loss)
        episode_lengths.append(move_count)
        max_tiles.append(max_tile)

        if max_tile in overview:
            overview[max_tile] += 1
        else:
            overview[max_tile] = 1

        print(f'\n--- max tile reached: {max_tile} --- move count: {move_count} --- average loss: {avg_loss}\n\n#########\n')
        with open(f"{saving_path}overview_training.txt", mode="w") as f:
            f.write(build_training_overview(overview))
        f.close()
        model.save(f"{saving_path}MCTS_CNN_2048.pt")

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