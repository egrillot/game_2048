from game_2048 import env_2048

def test_game():

    env = env_2048()
    print(env.grid)

    while not env.done:

        choosen_action = input('Action : ')
        if choosen_action == 'q':
            action = 0
        if choosen_action == 'z':
            action = 1
        if choosen_action == 'd':
            action = 2
        if choosen_action == 's':
            action = 3
        env.step(action)
                
        print(env.done, env.score)
        print(env.grid)