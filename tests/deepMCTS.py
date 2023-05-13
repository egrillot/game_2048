from game_2048.agent.alpha_zero.MCTS import MCTS, Node
from game_2048.agent.alpha_zero.State import State
from game_2048.agent.alpha_zero.NeuralNet import NeuralNet
from game_2048.agent.alpha_zero.NaiveAgent import NaiveAgent
from game_2048.env import env_2048

import numpy as np

nn = NeuralNet()

def test_mcts_selection():

    grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 4],
        [0, 0, 0, 0]
    ])
    state = State(grid)
    root = Node(state)
    for move in range(4):
        child_state, _ = state.play_move(move)
        child = Node(child_state, parent=root, policy=0.25)
        root.children.append(child)

    root.children[0].visit_count = 10
    root.children[1].visit_count = 20
    root.children[2].visit_count = 30
    root.children[3].visit_count = 40
    root.visit_count = 10 + 20 + 30 + 40

    mcts = MCTS(nn, num_simulations=100)
    selected_node = mcts.selection(root)

    assert selected_node == root.children[0]

def test_mcts_expansion():

    grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 2],
        [0, 0, 0, 0]
    ])
    state = State(grid)
    root = Node(state)
    mcts = MCTS(nn, num_simulations=100)
    value = mcts.expansion(root)

    assert len(root.children) == 4
    assert -1.0 <= value <= 1.0

def test_mcts_backpropagation():

    grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 2],
        [0, 0, 0, 0]
    ])
    state = State(grid)
    mcts = MCTS(nn, num_simulations=100)
    
    root = Node(state)
    state, _ = state.play_move(0)
    child1 = Node(state, parent=root, policy=0.25)
    root.children.append(child1)
    root.visit_count = 1

    state, _ = child1.state.play_move()
    child2 = Node(state, parent=child1, policy=1.0)
    child1.children.append(child2)
    child1.visit_count = 1

    state, _ = child2.state.play_move(0)
    child3 = Node(state, parent=child2, policy=1.0)
    child2.children.append(child3)

    state, _ = child3.state.play_move()
    child4 = Node(state, parent=child3, policy=1.0)
    child3.children.append(child4)

    state, _ = child4.state.play_move(child4.state.get_legal_moves()[0])
    child5 = Node(state, parent=child4, policy=1.0)
    child4.children.append(child5)

    value = 1.0
    mcts.backpropagate(value, child5)

    assert [root.visit_count, child1.visit_count, child2.visit_count, child3.visit_count, child4.visit_count, child5.visit_count] == [1, 2, 1, 1, 1, 1]
    assert [root.value, child1.value, child2.value, child3.value, child4.value, child5.value] == [0, 0.5, 1.0, 1.0, 1.0, 1.0]

def test_mcts_run():

    grid = np.array([
        [2, 4, 2, 4],
        [4, 2, 4, 2],
        [2, 4, 2, 2],
        [0, 0, 0, 0]
    ])
    state = State(grid)
    mcts = MCTS(nn, num_simulations=100)
    root = Node(state)

    mcts.run(root)

    assert sum([child.visit_count for child in root.children]) == 99
    assert abs(sum([child.policy for child in root.children]) - 1) < 1e-4

def test_deep_mcts_train():

    env = env_2048()
    env.reset()
    naive_agent = NaiveAgent()
    mcts = MCTS(naive_agent, 4)
    state = State(env.grid)
    current_node = Node(state)

    for _ in range(10):
        if current_node.state.player == 1:
            current_node_hash = current_node.signature()
            mcts.run(current_node)
            
            assert current_node_hash == current_node.signature()

            if sum([child.visit_count for child in current_node.children]) != 0:
                policy_target = np.zeros(4)
                for move in range(4):
                    childs = [child.visit_count for child in current_node.children if child.state.last_move == move]
                    policy_target[move] = childs[0] if len(childs) > 0 else 0
                
                policy_target /= policy_target.sum()

            else:
                policy_target = np.ones(4) / 4
            
            max_p = 0.0
            moves = [child.state.last_move for child in current_node.children]
            for i, p in enumerate(policy_target):
                if p >= max_p and i in moves:
                    best_move = i 
                    max_p = p
            
            current_node = [child for child in current_node.children if child.state.last_move == best_move][0]

        else:
            state, _ = current_node.state.play_move()
            child_node = Node(state, parent=current_node, policy=1.0)
            current_node.children.append(child_node)
            current_node = child_node

        print(current_node)

def test_deep_mcts():

    test_mcts_selection()
    test_mcts_expansion()
    test_mcts_backpropagation()
    test_mcts_run()
    test_deep_mcts_train()
