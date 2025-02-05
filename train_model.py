import time
import torch
import numpy as np
from rich.live import Live
from rich.console import Console
from rich.progress import Progress

from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random
import argparse
import math

# Training parameters
EPISODES = 300  # Increased episodes for deeper learning
EVAL_FREQUENCY = 20
EVAL_EPISODES = 200  # More evaluation games for better accuracy
BATCH_SIZE = 2048  # Larger batch size for more stable updates
TARGET_UPDATE = 100  # Less frequent updates to avoid instability
MEMORY_SIZE = 2000000  # Larger memory for better replay diversity
LEARNING_RATE = 0.000025  # Further reduced learning rate for smoother training
EPSILON_START = 1.0
EPSILON_END = 0.1  # Maintain more exploration for longer
EPSILON_DECAY = 0.99995  # Even slower decay for sustained exploration
GAMMA = 0.999  # Higher discount factor for better long-term planning
TAU = 0.001  # More gradual target network updates
GRADIENT_CLIP = 0.5  # Reduce large gradient spikes
PRIORITY_EPSILON = 1e-6  # Prevent NaN in priority-based sampling


def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes to train')
    parser.add_argument('--eval-frequency', type=int, default=EVAL_FREQUENCY, help='Evaluation frequency')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for replay')
    parser.add_argument('--eval-games', type=int, default=EVAL_EPISODES, help='Number of evaluation games')
    return parser.parse_args()


class CheckersTrainer:
    def __init__(self, env, agent1, agent2, args):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = args.episodes
        self.eval_frequency = args.eval_frequency
        self.eval_games = args.eval_games
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.rewards = []
        self.win_rates = []
        self.losses = []
        self.epsilons = []

    def train(self):
        wins = {1: 0, -1: 0, 0: 0}
        os.makedirs('checkpoints', exist_ok=True)
        console = Console()
        with Live(console=console, refresh_per_second=2):
            for episode in range(self.episodes):
                state = self.env.reset()
                done = False
                current_agent, opponent_agent = (self.agent1, self.agent2) if random.random() < 0.5 else (self.agent2, self.agent1)
                opponent_agent.epsilon = max(EPSILON_END, EPSILON_START - (episode / self.episodes) * (EPSILON_START - EPSILON_END))
                player = 1 if current_agent == self.agent1 else -1
                total_reward = 0
                total_loss = 0
                move_count = 0

                while not done:
                    valid_moves = self.env.valid_moves(player)
                    if not valid_moves:
                        break

                    action = current_agent.act(state, valid_moves)
                    next_state, reward, additional_moves, done = self.env.step(action, player)

                    reward = max(-2, min(reward, 2))
                    reward += 0.1 * (action[2] - action[0]) if player == 1 else 0.1 * (action[0] - action[2])
                    reward += 2.0 if abs(action[2] - action[0]) == 2 else 0.0
                    reward += 4.0 if (player == 1 and action[2] == self.env.board_size - 1) or (player == -1 and action[2] == 0) else 0.0

                    current_agent.remember(state, action, reward, next_state, done)
                    if len(current_agent.memory) > current_agent.batch_size:
                        loss = current_agent.replay()
                        if loss is not None:
                            total_loss += loss
                        if current_agent.epsilon > current_agent.epsilon_min:
                            current_agent.epsilon = max(current_agent.epsilon_min, current_agent.epsilon * current_agent.epsilon_decay)

                    state = next_state
                    total_reward += reward
                    move_count += 1
                    if not additional_moves:
                        current_agent, opponent_agent = opponent_agent, current_agent
                        player *= -1

                if episode % TARGET_UPDATE == 0:
                    self.agent1.update_target_network()
                    self.agent2.update_target_network()

                self.rewards.append(total_reward)
                self.losses.append(total_loss / max(1, move_count))
                self.epsilons.append(current_agent.epsilon)
                wins[self.env.game_winner(state)] += 1

                if (episode + 1) % self.eval_frequency == 0:
                    win_data = self.evaluate()
                    win_rate = (win_data[1] / self.eval_games) * 100
                    self.win_rates.append(win_rate)
                    print(f"Episode {episode + 1}: Win rate {win_rate:.2f}%")

        self.save_model('checkpoints/model_final.pth')
        self.plot_training_results()
        return self.agent1, self.agent2, self.rewards, self.win_rates

    def evaluate(self):
        wins = {1: 0, -1: 0, 0: 0}
        for _ in range(self.eval_games):
            state = self.env.reset()
            done, player = False, 1
            while not done:
                agent = self.agent1 if player == 1 else self.agent2
                valid_moves = self.env.valid_moves(player)
                if not valid_moves:
                    wins[0] += 1
                    break
                action = agent.act(state, valid_moves)
                state, _, additional_moves, done = self.env.step(action, player)
                if not additional_moves:
                    player *= -1
            wins[self.env.game_winner(state)] += 1
        return wins

    def plot_training_results(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.win_rates, label='Win Rate')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate Progression')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.rewards, label='Rewards per Episode', color='green')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Training Rewards')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        best_agent = self.agent1 if self.win_rates[-1] > self.win_rates[-2] else self.agent2
        torch.save(best_agent.q_network.state_dict(), path)
        print(f"Model saved to {path} with win rate: {self.win_rates[-1]:.2f}%")


def train_agent():
    env = checkers_env()
    agent1, agent2 = DQNAgent(), DQNAgent()
    trainer = CheckersTrainer(env, agent1, agent2, parse_args())
    return trainer.train()


if __name__ == "__main__":
    train_agent()
