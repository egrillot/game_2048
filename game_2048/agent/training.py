from .baseline import Agent
from ..env import env_2048
from ..utils import ProgressBar

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt

actions = {
    0: 'left',
    1: 'up',
    2: 'right',
    3: 'down'
}

def espilon_greedy_search(
    agent: Agent,
    max_iteration: int,
    epochs: int,
    epsilon: float,
    decrease_function: Callable,
    env: env_2048,
    verbose=1
) -> None:
    
    if epsilon < 0 or epsilon > 1:
        raise ValueError(f"The provided epsilon value is equal to {epsilon} instead of being between [0, 1].")

    print(f"Training the agent {agent.get_name()}\nwith parameters : {agent.get_descritpion()}\n")
    print(f"Epsilon greedy search parameters : max_iteration={max_iteration}, epochs={epochs} and epsilon={epsilon}.\n")

    all_epochs_rewards = []
    all_history = []
    max_scores = []
    step_count = 1
    mean_loss = []

    for epoch in range(1, epochs + 1):
                    
        print(f"Epoch : {epoch}/{epochs}, epsilon = {epsilon}")

        env.reset()
        total_reward = 0.0
        all_rewards = []

        history = {
            'exploit': {
                'left': {
                    'successfull': 0,
                    'failed': 0
                },
                'up': {
                    'successfull': 0,
                    'failed': 0
                },
                'right': {
                    'successfull': 0,
                    'failed': 0
                },
                'down': {
                    'successfull': 0,
                    'failed': 0
                }
            },
            'explore': {
                'left': {
                    'successfull': 0,
                    'failed': 0
                },
                'up': {
                    'successfull': 0,
                    'failed': 0
                },
                'right': {
                    'successfull': 0,
                    'failed': 0
                },
                'down': {
                    'successfull': 0,
                    'failed': 0
                }
            },
            'exploit -> explore': 0
        }

        successfull_action = 0
        failed_action = 0

        bar = ProgressBar(target=max_iteration)

        for iteration in range(1, max_iteration + 1):

            p = np.random.random()

            if p <= epsilon:

                action = agent.explore(env)
                action_type = 'explore'

            else:

                performable, action = agent.exploit(env)
                action_type = 'exploit'

                if not performable:
                    action = agent.explore(env)
                    action_type = 'exploit -> explore'

            grid, next_grid, env_done, reward, _ = env.step(action)

            if reward <= 0:

                result = 'failed'
                failed_action += 1

            else :

                result = 'successfull'
                successfull_action += 1

            if action_type != 'exploit -> explore':

                history[action_type][actions[action]][result] += 1
            
            else:

                history['exploit -> explore'] += 1     

            agent.learn(grid, next_grid, action, reward, env_done)
            total_reward += reward
            loss = agent.loss()
            exploit_count = sum([sum(list(items.values())) for items in history['exploit'].values()])
            explore_count = sum([sum(list(items.values())) for items in history['explore'].values()])

            if verbose > 0:

                if verbose == 1:

                    values = {
                        'cumulate rewards': total_reward,
                        'max reached': np.max(next_grid),
                        'sucessfull actions count': successfull_action,
                        'failed actions count': failed_action,
                        'explore count': history['exploit -> explore'] + explore_count,
                        'exploit count': exploit_count,
                        'model loss': np.mean(loss) if loss is not None else None
                    }
                
                if verbose == 2:

                    values = {
                        'cumulate rewards': total_reward,
                        'sucessfull action': successfull_action,
                        'failed action': failed_action,
                        'explore count': history['exploit -> explore'] + explore_count,
                        'exploit count': exploit_count,
                        'exploit successfull': sum([items['successfull'] for items in history['exploit'].values()]),
                        'exploit failed': sum([items['failed'] for items in history['exploit'].values()]),
                        'model loss': np.mean(loss) if loss is not None else None,
                        'max reached': np.max(next_grid)
                    }

                bar.update(values, iteration-1)

            all_rewards.append(reward)
            if env_done:

                break
            
            epsilon = decrease_function(epsilon, step_count)
            step_count += 1

        all_epochs_rewards.append(all_rewards)
        max_scores.append(np.max(next_grid))
        all_history.append(history)
        mean_loss.append(np.mean(loss) if loss is not None else None)
        agent.new_episode()

        if verbose > 0:

            bar.update(values, max_iteration)

        exploit_count = sum([sum(list(items.values())) for items in history['exploit'].values()])

        print(f"\nEpoch ended at {iteration} iterations - Exploit deflected to explore count : {history['exploit -> explore']} - Exploit count : {exploit_count}. \n\n###################\n")

    lengths = [len(epoch) for epoch in all_epochs_rewards]
    longest_epoch_length = max(lengths)
    all_cumulatives_reward = np.zeros((epochs, longest_epoch_length), dtype=float)

    fig, ax = plt.subplots(3, 2, figsize=(30, 15))
    fig.suptitle(f"Involved agent : {agent.get_name()}, with parameters : max_iteration={max_iteration}, epochs={epochs} and epsilon={epsilon}.", fontsize=14)

    for i, episode in enumerate(all_epochs_rewards):
        
        episode_length = len(episode)
        episode = np.array(episode)
        paded_episode = np.pad(episode, pad_width=(0, longest_epoch_length - episode_length))
        all_cumulatives_reward[i, :] = np.cumsum(paded_episode)

    avg = np.average(all_cumulatives_reward, axis=0)
    std = np.std(all_cumulatives_reward, axis=0)

    x = [i for i in range(longest_epoch_length)]
    ax[0, 0].plot(x, avg)
    ax[0, 0].fill_between(x, avg - std, avg + std, alpha=0.5)
    ax[0, 0].set_title('Cumulative rewards vs iterations')
    ax[0, 0].set(xlabel='iteration', ylabel='cumulative reward')

    x = [i for i in range(epochs)]
    ax[0, 1].plot(x, lengths)
    ax[0, 1].set_title('Duration vs epochs')
    ax[0, 1].set(xlabel='epoch', ylabel='duration')

    all_history_count = np.zeros((epochs, 4), dtype=int)

    for i, history in enumerate(all_history):

        all_history_count[i, 0] = history['exploit']['left']['successfull'] + history['exploit']['up']['successfull'] + history['exploit']['right']['successfull'] + history['exploit']['down']['successfull']
        all_history_count[i, 1] = history['exploit']['left']['failed'] + history['exploit']['up']['failed'] + history['exploit']['right']['failed'] + history['exploit']['down']['failed']
        all_history_count[i, 2] = history['explore']['left']['successfull'] + history['explore']['up']['successfull'] + history['explore']['right']['successfull'] + history['explore']['down']['successfull']
        all_history_count[i, 3] = history['explore']['left']['failed'] + history['explore']['up']['failed'] + history['explore']['right']['failed'] + history['explore']['down']['failed']

    ax[1, 0].plot(x, all_history_count[:, 0], label='successfull exploit')
    ax[1, 0].plot(x, all_history_count[:, 1], label='failed exploit')
    ax[1, 0].plot(x, all_history_count[:, 2], label='successfull explore')
    ax[1, 0].plot(x, all_history_count[:, 3], label='failed explore')
    ax[1, 0].set_title('Success & failed action count by exploration and exploitation')
    ax[1, 0].set(xlabel='epoch', ylabel='count')
    ax[1, 0].legend(loc='lower right')

    ax[1, 1].scatter(x, max_scores, label='max score')
    ax[1, 1].set_title('Max score vs epoch')
    ax[1, 1].set(xlabel='epoch', ylabel='rate')
    ax[1, 1].legend(loc='lower right')

    ax[2, 0].plot(x, mean_loss)
    ax[2, 0].set_title('Mean loss vs epoch')
    ax[2, 0].set(xlabel='epoch', ylabel='mean loss')

    ax[2, 1].plot(x, [history['exploit -> explore'] for history in all_history])
    ax[2, 1].set_title('Exploit to explore vs epoch')
    ax[2, 1].set(xlabel='epoch', ylabel='deflection count')

    plt.show()