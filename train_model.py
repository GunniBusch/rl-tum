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
EPISODES = 3000
EVAL_FREQUENCY = 50
EVAL_EPISODES = 50
BATCH_SIZE = 512
TARGET_UPDATE = 5  # More frequent updates
MEMORY_SIZE = 500000
LEARNING_RATE = 0.0002
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.999  # Slower decay for better exploration
GAMMA = 0.99  # Long-term reward optimization
TAU = 0.002  # More gradual target network updates
GRADIENT_CLIP = 1.0  # Prevent large gradient updates


def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=3000, help='Number of episodes to train')
    parser.add_argument('--eval-frequency', type=int, default=50, help='Evaluation frequency')
    parser.add_argument('--learning-rate', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for replay')
    parser.add_argument('--eval-games', type=int, default=50, help='Number of evaluation games')
    return parser.parse_args()


class CheckersTrainer:
    def __init__(self, env, agent1, agent2, args):
        self.env = env
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = args.episodes
        self.eval_frequency = args.eval_frequency
        self.eval_games = args.eval_games
        self.args = args
        self.rewards = []
        self.win_rates = []

    def train(self):
        wins = {1: 0, -1: 0, 0: 0}
        os.makedirs('checkpoints', exist_ok=True)
        console = Console()
        with Live(console=console, refresh_per_second=2):
            for episode in range(self.episodes):
                state = self.env.reset()
                done = False
                current_agent, opponent_agent = (self.agent1, self.agent2) if random.random() < 0.5 else (self.agent2, self.agent1)
                opponent_agent.epsilon = max(0.01, 0.2 - (episode / self.episodes) * 0.15)
                player = 1 if current_agent == self.agent1 else -1
                total_reward = 0

                while not done:
                    valid_moves = self.env.valid_moves(player)
                    if not valid_moves:
                        break

                    action = current_agent.act(state, valid_moves)
                    next_state, reward, additional_moves, done = self.env.step(action, player)

                    reward = max(-10, min(reward, 10))  # Clip rewards for stability
                    reward += 0.5 * (action[2] - action[0]) if player == 1 else 0.5 * (action[0] - action[2])
                    reward += 6.0 if abs(action[2] - action[0]) == 2 else 0.0
                    reward += 8.0 if (player == 1 and action[2] == self.env.board_size - 1) or (player == -1 and action[2] == 0) else 0.0

                    current_agent.remember(state, action, reward, next_state, done)
                    if len(current_agent.memory) > BATCH_SIZE:
                        current_agent.replay()
                        current_agent.epsilon = max(current_agent.epsilon_min, current_agent.epsilon * EPSILON_DECAY)

                    state = next_state
                    total_reward += reward
                    if not additional_moves:
                        current_agent, opponent_agent = opponent_agent, current_agent
                        player *= -1

                if episode % TARGET_UPDATE == 0:
                    self.agent1.update_target_network()
                    self.agent2.update_target_network()

                self.rewards.append(total_reward)
                wins[self.env.game_winner(state)] += 1

                if (episode + 1) % self.eval_frequency == 0:
                    win_rate = (self.evaluate()[1] / max(1, sum(wins.values()))) * 100
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
                if not valid_moves and not self.env.valid_moves(-player):
                    wins[0] += 1
                    break
                action = agent.act(state, valid_moves) if valid_moves else None
                state, _, additional_moves, done = self.env.step(action, player) if action else (state, 0, False, True)
                if not additional_moves:
                    player *= -1
            wins[self.env.game_winner(state)] += 1
        return wins

    def plot_training_results(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.win_rates, label='Win Rate')
        plt.xlabel('Evaluation Step')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate Progression')
        plt.legend()
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
