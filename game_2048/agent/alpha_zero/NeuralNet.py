from .State import State

from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import torch.cuda
from torch.nn.utils.clip_grad import clip_grad_norm_

from typing import Tuple, List
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        self.policy = nn.Linear(128, 4)
        self.value = nn.Linear(128, 1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:

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

        return self.policy(x), self.value(x)
    
class NeuralNet:

    def __init__(self, model=None, learning_rate=0.001, weight_decay=1e-4, training=True) -> None:
        self.wrapper = Wrapper()
        self.model = model if model else Model().to(device)
        if training:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.training = training

    def predict(self, state: State) -> Tuple[np.ndarray, float]:
        wrapped_grid = self.wrapper.get_state(state.grid)
        wrapped_grid = torch.from_numpy(wrapped_grid).unsqueeze(0).float().to(device)
        with torch.no_grad():
            policy_logits, value = self.model(wrapped_grid)
        policy = np.array(F.softmax(policy_logits, dim=1).cpu().tolist()[0])
        value = torch.tanh(value).tolist()[0][0]

        return policy, value
    
    def train(self, states: List[State], policies: List[np.ndarray], values: List[float], batch_size=64, epochs=1) -> float:
        if not self.training:
            raise ValueError("This instance of NeuralNet is not set up for training.")
        
        n = len(states)
        wrapped_states = [self.wrapper.get_state(state.grid) for state in states]
        wrapped_states = torch.tensor(np.array(wrapped_states), device=device).float()
        target_policies = torch.tensor(np.array(policies), device=device).float()
        target_values = torch.tensor(np.array(values), device=device).float()
        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            permutation = torch.randperm(n)

            for i in range(0, n, batch_size):
                indices = permutation[i: i+ batch_size]
            
                batch_states = wrapped_states[indices]
                batch_target_policies = target_policies[indices]
                batch_target_values = target_values[indices]

                self.optimizer.zero_grad()

                pred_policies, pred_values = self.model(batch_states)
                policy_loss = F.kl_div(F.log_softmax(pred_policies, dim=1), batch_target_policies, reduction='batchmean')
                value_loss = F.mse_loss(pred_values.squeeze(-1), batch_target_values)

                loss = policy_loss + value_loss
                total_loss += loss.item()
                num_batches += 1
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
        
        return total_loss / num_batches

    def save(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def load(self, file_path):
        state_dict = torch.load(file_path)
        self.model.load_state_dict(state_dict)