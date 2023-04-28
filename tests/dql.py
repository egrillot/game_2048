from game_2048.env import env_2048
from game_2048.agent import DeepQlearner

def test_dql_exploit():

    env = env_2048()
    agent = DeepQlearner('deepQlearner', 1000, 10)
    print(agent.exploit(env))