from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.multiprocessing as mp
import torch.nn.functional as F
from numpy import dtype
from numpy.ma.core import indices
from torchrl.data import ReplayBuffer, ListStorage, PrioritizedReplayBuffer, PrioritizedSampler, TensorStorage, \
    LazyTensorStorage

mp.set_start_method('spawn', force=True)  # Add this at the top of the file




class ParallelDQN(nn.Module):
    def __init__(self, state_size, action_size=36, hidden_size=128, dropout_rate=0.1):
        super(ParallelDQN, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation on output


class ParallelDQN2(nn.Module):
    def __init__(self, state_size, action_size=36, hidden_size=128, dropout_rate=0.1):
        super(ParallelDQN2, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Helps with generalization

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Dropout again

            nn.Linear(hidden_size, action_size)  # Output raw Q-values

        )

    def forward(self, x):
        return self.model(x)

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.memory = deque(maxlen=capacity)
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.priority_epsilon = 1e-6  # Add this here

    def __len__(self):  # Add this method
        return len(self.memory)

    def push(self, state, action, reward, next_state, done):
        # New experiences get max priority
        max_priority = np.max(self.priorities) if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.memory) == 0:
            return [], [], []

        # Calculate sampling probabilities safely
        priorities = self.priorities[:len(self.memory)] + self.priority_epsilon
        probs = priorities ** self.alpha
        probs /= probs.sum() if probs.sum() > 0 else np.ones_like(probs)  # Prevent NaN

        # Sample indices based on priorities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)

        # Calculate importance sampling weights
        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max() if weights.max() > 0 else 1  # Prevent NaN

        samples = [self.memory[idx] for idx in indices]
        return samples, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + self.priority_epsilon

class DQNAgent:
    def __init__(self, state_size=36, action_size=1296):
        # Set random seeds for reproducibility
        #torch.manual_seed(42)
        #np.random.seed(42)
        #random.seed(42)
        # if torch.cuda.is_available():
        # torch.cuda.manual_seed_all(42)

        self.state_size = state_size
        self.action_size = action_size
        # Add priority replay parameters
        self.priority_alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.priority_beta = 0.4   # Importance sampling correction (starts low, annealed to 1)
        self.priority_epsilon = 1e-6  # Small constant to prevent zero priorities

        self.memory = PriorityReplayBuffer(100000, 0.6)

        # Enhanced training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.0001  # Minimum exploration rate
        self.epsilon_decay = 0.999  # More gradual decay (was 0.995)
        self.learning_rate = 0.001
        self.batch_size = 128  # Increased batch size for H100
        self.hidden_size = 128
        # H100 specific optimizations
        if torch.cuda.is_available():

            self.device = torch.device("cuda")

            # Enable TF32 and other optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.deterministic = True  # For reproducibility

            # Set higher memory fraction for GPU
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory

            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"Memory Usage:")
            print(f"Allocated: {torch.cuda.memory_allocated(0)//1024//1024}MB")
            print(f"Cached: {torch.cuda.memory_reserved(0)//1024//1024}MB")
        else:
            self.device = torch.device("cpu")
            print("WARNING: No GPU found, using CPU")

        def create_network():
            model = nn.Sequential(
                nn.Linear(state_size, 1024),  # Wider network
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, action_size)
            )

            model = model.to(self.device)
            if torch.cuda.device_count() > 1:
                print(f"Using {torch.cuda.device_count()} GPUs!")
                return nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            return model

        # Create networks and ensure they're on GPU
        self.q_network = ParallelDQN(state_size, action_size, dropout_rate=0.2).to(self.device)
        self.target_network = ParallelDQN(state_size, action_size, dropout_rate=0.1).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        # self.memory = PrioritizedReplayBuffer(storage=LazyTensorStorage(100000, device=self.device), alpha=self.priority_alpha,beta=self.priority_beta,eps=self.priority_epsilon, pin_memory=True)

        # Use mixed precision training only if CUDA is available
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        # Move optimizer to GPU
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)



    def remember(self, state, action, reward, next_state, done):
        # Store as numpy arrays
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, valid_moves):
        if torch.rand(1).item() < self.epsilon:
            return random.choice(valid_moves)

        # Set the network to evaluation mode
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                np.asarray(state, dtype=np.float32).flatten()
            ).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor).cpu().numpy().flatten()

        # Restore the network to training mode
        self.q_network.train()

        # Compute Q-values for valid moves only
        valid_q_values = [q_values[self.encode_action(move)] for move in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        return valid_moves[best_move_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            print(f"Memory too small: {len(self.memory)} / {self.batch_size}")
            return None

        # Sample a batch from memory. Each element in batch is assumed to be
        # (state, action, reward, next_state, done)
        batch, indices, weights = self.memory.sample(self.batch_size, self.priority_beta)

        # Prepare batch tensors
        states = torch.FloatTensor(
            np.array([np.asarray(s, dtype=np.float32).flatten() for s, _, _, _, _ in batch])
        ).to(self.device)

        actions = torch.LongTensor(
            [self.encode_action(a) for _, a, _, _, _ in batch]
        ).to(self.device)

        # Ensure rewards are scalars. If each reward is a vector, sum its components.
        rewards = torch.FloatTensor(
            [np.sum(r) if isinstance(r, (list, np.ndarray)) else r for _, r, _, _, _ in batch]
        ).to(self.device)

        next_states = torch.FloatTensor(
            np.array([np.asarray(ns, dtype=np.float32).flatten() for _, _, _, ns, _ in batch])
        ).to(self.device)

        # Convert done flags to floats: 1.0 if done, else 0.0
        dones = torch.FloatTensor(
            [1.0 if done else 0.0 for _, _, _, _, done in batch]
        ).to(self.device)

        weights = torch.FloatTensor(weights).to(self.device)

        # Set networks to proper modes
        self.q_network.train()
        self.target_network.eval()

        # Use mixed precision if CUDA is available
        autocast_context = torch.amp.autocast('cuda') if torch.cuda.is_available() else nullcontext
        with autocast_context:
            # Current Q-values for all actions in the current states
            current_q_values = self.q_network(states)  # shape: (batch_size, num_actions)
            # Pick Q-values corresponding to taken actions
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Compute target Q-values using the target network
            with torch.no_grad():
                next_q_values = self.target_network(next_states)  # shape: (batch_size, num_actions)
                max_next_q, _ = next_q_values.max(dim=1)
                target_q = rewards + (1 - dones) * self.gamma * max_next_q

            # Compute weighted MSE loss
            loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad()
        if torch.cuda.is_available():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update epsilon for epsilon-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # Compute TD errors for priority update and update the replay memory
        td_errors = torch.abs(target_q - current_q).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        return loss.item()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos