import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import torch.multiprocessing as mp
import torch.nn.functional as F

mp.set_start_method('spawn', force=True)  # Add this at the top of the file



class ParallelDQN(nn.Module):
    def __init__(self, state_shape=(6, 6), action_size=36, hidden_size=64, dropout_rate=0.2):
        super(ParallelDQN, self).__init__()

        # Minimal Convolutional Feature Extractor
        self.conv = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)

        # Flattened representation
        self.fc1 = nn.Linear(16 * 6 * 6, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)  # Prevents overfitting

        # Dueling DQN: Value and Advantage branches
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = x.view(-1, 1, 6, 6)  # Reshape input into (batch, channels, height, width)
        x = F.relu(self.conv(x))  # Simple feature extraction

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.layer_norm(x)  # Normalization to avoid overfitting
        x = self.dropout(x)  # Prevents co-adaptation

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Dueling Q-value formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

    def select_action(self, state, epsilon=0.1):
        """ Uses ε-greedy action selection (simpler than Noisy Networks) """
        if np.random.rand() < epsilon:
            return np.random.randint(0, 36)  # Random move for exploration
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()  # Greedy action


class ParallelDQNB(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=512):
        super(ParallelDQNB, self).__init__()

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)

        self.dropout = nn.Dropout(0.2)

        # Enable parallel processing for multiple GPUs if available
        if torch.cuda.device_count() > 1:
            self = nn.DataParallel(self)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)

class PriorityReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use
        self.memory = []
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
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.state_size = state_size
        self.action_size = action_size
        self.memory = PriorityReplayBuffer(100000, 0.6)

        # Enhanced training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Starting exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.9995  # More gradual decay (was 0.995)
        self.learning_rate = 0.0001
        self.batch_size = 128  # Increased batch size for H100
        self.hidden_size = 128
        # H100 specific optimizations
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = torch.device("cuda:0")

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
        self.q_network = ParallelDQN(state_size, action_size, self.hidden_size).to(self.device)
        self.target_network = ParallelDQN(state_size, action_size, self.hidden_size).to(self.device)

        # Use mixed precision training only if CUDA is available
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        # Move optimizer to GPU
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

        # Add priority replay parameters
        self.priority_alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.priority_beta = 0.4   # Importance sampling correction (starts low, annealed to 1)
        self.priority_epsilon = 1e-6  # Small constant to prevent zero priorities

    def remember(self, state, action, reward, next_state, done):
        # Store as numpy arrays
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        self.memory.push(state, action, reward, next_state, done)

    def act(self, state, valid_moves):
        if random.random() <= self.epsilon:
            return random.choice(valid_moves)

        # Set to eval mode for prediction
        self.q_network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(np.asarray(state, dtype=np.float32).flatten()).unsqueeze(0).to(self.device)
            action_values = self.q_network(state).cpu()

        # Filter only valid moves
        valid_q_values = [action_values[0][self.encode_action(move)] for move in valid_moves]
        best_move_idx = np.argmax(valid_q_values)
        return valid_moves[best_move_idx]

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch, indices, weights = self.memory.sample(self.batch_size, self.priority_beta)
        weights = torch.FloatTensor(weights).to(self.device, non_blocking=True)

        # Prepare batch data efficiently
        states = torch.from_numpy(np.vstack([s.flatten() for s, _, _, _, _ in batch])).float()
        next_states = torch.from_numpy(np.vstack([ns.flatten() for _, _, _, ns, _ in batch])).float()

        # Handle data transfer based on device
        if torch.cuda.is_available():
            # Use CUDA streams for parallel data transfer
            with torch.cuda.stream(torch.cuda.Stream()):
                states = states.pin_memory().to(self.device, non_blocking=True)
                next_states = next_states.pin_memory().to(self.device, non_blocking=True)
        else:
            # CPU path - simple transfer
            states = states.to(self.device)
            next_states = next_states.to(self.device)

        self.q_network.train()
        self.target_network.eval()

        # Use mixed precision training only if CUDA is available
        if torch.cuda.is_available():
            with torch.amp.autocast('cuda'):
                current_q_values = self.q_network(states)
                with torch.no_grad():
                    next_q_values = self.target_network(next_states)

                target = current_q_values.clone()
                td_errors = []

                for i, (_, action, reward, _, done) in enumerate(batch):
                    if done:
                        target_value = reward
                    else:
                        target_value = reward + self.gamma * torch.max(next_q_values[i])

                    current_value = current_q_values[i][self.encode_action(action)]
                    td_error = abs(target_value - current_value.item())
                    td_errors.append(td_error)

                    target[i][self.encode_action(action)] = target_value

                losses = F.mse_loss(current_q_values, target, reduction='none')
                weighted_loss = (weights.unsqueeze(1) * losses.mean(dim=1)).mean()

            self.optimizer.zero_grad()
            self.scaler.scale(weighted_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # CPU training path
            current_q_values = self.q_network(states)
            with torch.no_grad():
                next_q_values = self.target_network(next_states)

            target = current_q_values.clone()
            td_errors = []

            for i, (_, action, reward, _, done) in enumerate(batch):
                if done:
                    target_value = reward
                else:
                    target_value = reward + self.gamma * torch.max(next_q_values[i])

                current_value = current_q_values[i][self.encode_action(action)]
                td_error = abs(target_value - current_value.item())
                td_errors.append(td_error)

                target[i][self.encode_action(action)] = target_value

            losses = F.mse_loss(current_q_values, target, reduction='none')
            weighted_loss = (weights.unsqueeze(1) * losses.mean(dim=1)).mean()

            self.optimizer.zero_grad()
            weighted_loss.backward()
            self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def encode_action(self, action):
        """Convert action [start_row, start_col, end_row, end_col] to index"""
        start_pos = action[0] * 6 + action[1]
        end_pos = action[2] * 6 + action[3]
        return start_pos * 6 + end_pos