from .baseline import Agent
from ..env import env_2048

from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.clip_grad import clip_grad_norm_

from typing import NamedTuple, Union, Tuple, List
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Transition(NamedTuple):

    state: Union[Tuple[Tensor], List[Tensor]]
    action: Union[Tuple[Tensor], List[Tensor]]
    next_state: Union[Tuple[Tensor], List[Tensor]]
    reward: Union[Tuple[Tensor], List[Tensor]]

class ReplayMemory:

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Model(nn.Module):

    def __init__(self, dropout_rate=0.2) -> None:
        super().__init__()

        self.conv1a = nn.Conv2d(16, 64, kernel_size=2, padding=0)
        self.conv1b = nn.Conv2d(16, 64, kernel_size=3, padding=1)

        self.conv2a = nn.Conv2d(64, 128, kernel_size=2)
        self.conv2b = nn.Conv2d(64, 128, kernel_size=3)
        self.conv2c = nn.Conv2d(64, 128, kernel_size=3)
        self.conv2d = nn.Conv2d(64, 128, kernel_size=2)

        self.dropout = nn.Dropout(dropout_rate)

        self.linear1 = nn.Linear(2304, 128)
        self.linear2 = nn.Linear(128, 4)

    def forward(self, x: Tensor) -> Tensor:

        x1 = F.relu(self.conv1a(x))
        x2 = F.relu(self.conv1b(x))

        x1a = F.relu(self.conv2a(x1))
        x1b = F.relu(self.conv2b(x1))
        x2a = F.relu(self.conv2a(x2))
        x2b = F.relu(self.conv2b(x2))

        x1a = torch.flatten(x1a, start_dim=1)
        x1b = torch.flatten(x1b, start_dim=1)
        x2a = torch.flatten(x2a, start_dim=1)
        x2b = torch.flatten(x2b, start_dim=1)

        x_concat = torch.cat((x1a, x1b, x2a, x2b), dim=1)
        x_dropout = self.dropout(x_concat)

        x = F.relu(self.linear1(x_dropout))
        x = self.linear2(x)

        return x
    
class Wrapper:

    def __init__(self) -> None:
        pass

    def get_state(self, grid: np.ndarray) -> np.ndarray:
        wrapped_grid = np.zeros((16, 4, 4), dtype=float)
        grid_masked = np.where(grid > 0, grid, 1)
        log_grid = np.log2(grid_masked)
        non_zero_indices = np.argwhere(log_grid > 0)
        raveled_log_grid = np.ravel(log_grid)
        non_zero_values = raveled_log_grid[np.nonzero(raveled_log_grid)[0]].astype(int)
        wrapped_grid[non_zero_values - 1, non_zero_indices[:, 0], non_zero_indices[:, 1]] = 1.0

        return wrapped_grid

class DeepQlearner(Agent):

    def __init__(
        self, 
        name: str, 
        replay_memory_size=10000, 
        target_update=10, 
        gamma=0.015, 
        learning_rate=0.01, 
        clip_grad=1.0, 
        batch_size=512
    ) -> None:
        super().__init__(name)

        self.gamma = gamma
        self.lr = learning_rate
        self.clip_grad = clip_grad,
        self.batch_size = batch_size

        self.policy_net = Model().to(device)
        self.target_net = Model().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.target_update = target_update
        self.loss_value = None
        self.episode = 0

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayMemory(replay_memory_size)
        self.wrapper = Wrapper()

    def optimize_model(self, norm_clipping=False):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map((lambda s: s is not None), batch.next_state)),
                                      device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        output = self.policy_net(state_batch)

        state_action_values = output.gather(1, action_batch)
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.loss_value = np.array(loss.tolist())

        if norm_clipping:
            clip_grad_norm_(self.policy_net.parameters(), 1.0)
        else:
            for param in self.policy_net.parameters():
                if param.grad is not None:
                    param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def exploit(self, env: env_2048) -> Tuple[bool, int]:
        with torch.no_grad():
            state = self.wrapper.get_state(env.grid)
            tensor_grid = torch.tensor(state).to(device)
            predicted_rewards = self.policy_net(tensor_grid.float().unsqueeze(0)).max(1)

        choosen_action = predicted_rewards[1].tolist()[0]

        if env.is_action_valid(choosen_action):

            return True, choosen_action
        
        self.learn(
            reward=0.0,
            action=choosen_action,
            grid=state,
            next_grid=state,
            done=False
        )

        return False, choosen_action
    
    def explore(self, env: env_2048) -> int:
        return env.sample_valid_action()
    
    def learn(self, grid: np.ndarray, next_grid: np.ndarray, action: int, reward: float, done: bool) -> None:
        state = self.wrapper.get_state(grid)
        next_state = self.wrapper.get_state(next_grid)
        reward_tensor = torch.tensor([reward], device=device, dtype=torch.float)
        action_tensor = torch.tensor([[action]], device=device, dtype=torch.int64)
        state_tensor = torch.as_tensor(state, dtype=torch.float, device=device).unsqueeze(0)
        if done:
            next_state_tensor = None
        else:
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float, device=device).unsqueeze(0)
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor)

        self.optimize_model()

    def get_descritpion(self) -> str:
        return f"DQL: trainable parameters={sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)},\
            lr={self.lr}, target_update={self.target_update}, γ={self.gamma}, memory_length={self.memory.capacity},\
            batch_size={self.batch_size}"

    def save(self, directory_path: str) -> None:
        return torch.save(self.target_net.state_dict(), f"{directory_path}/ConvNet.pt")
    
    def loss(self) -> None:
        return self.loss_value
    
    def new_episode(self) -> None:
        self.loss_value = None
        self.episode += 1        

        if self.episode % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
