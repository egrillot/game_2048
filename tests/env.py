from game_2048 import env_2048 

import numpy as np

def test_left_action(nb_action=10):

    env = env_2048()
    print(env.grid)
    for _ in range(nb_action):
        env.step(action=0)
        print(env.grid)

def test_up_action(nb_action=10):

    env = env_2048()
    print(env.grid)
    for _ in range(nb_action):
        env.step(action=1)
        print(env.grid)

def test_right_action(nb_action=10):

    env = env_2048()
    print(env.grid)
    for _ in range(nb_action):
        env.step(action=2)
        print(env.grid)

def test_down_action(nb_action=10):

    env = env_2048()
    print(env.grid)
    for _ in range(nb_action):
        env.step(action=3)
        print(env.grid)

def test_actions(nb_action=50):

    env = env_2048()
    print(env.grid)
    for _ in range(nb_action):
        action = np.random.choice(4)
        env.step(action=action)
        print('action : ', action)
        print(env.grid)

def test_random_valid_action(nb_action=200):

    env = env_2048()
    print(env.grid)
    move_count = 1

    while not env.done and move_count < nb_action:
        action = env.sample_valid_action()
        env.step(action=action)
        print('action : ', action)
        print(env.grid)
        move_count += 1
