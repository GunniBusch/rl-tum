import time
import torch
import numpy as np
from rich.live import Live
from rich.console import Console
from rich.progress import Progress
from sympy.abc import alpha

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
EPSILON_END = 0.0001  # Maintain more exploration for longer
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
        self.eval_results = []

    def soft_update(self, target, source, tau=0.005):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def train(self):
        """ Main training loop for the agents """
        wins = {1: 0, -1: 0, 0: 0}
        checkpoint_dir = 'checkpoints'
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        console = Console()
        with Live(console=console, refresh_per_second=2):
            for episode in range(self.episodes):
                state = self.env.reset()
                done = False
                current_agent, opponent_agent = (self.agent1, self.agent2) if random.random() < 0.5 else (
                self.agent2, self.agent1)
                opponent_agent.epsilon = max(EPSILON_END, opponent_agent.epsilon * EPSILON_DECAY)
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
                    reward += 0.2 * (action[2] - action[0]) if player == 1 else 0.2 * (action[0] - action[2])
                    reward += 5.0 if abs(action[2] - action[0]) == 2 else 0.0
                    reward += 8.0 if (player == 1 and action[2] == self.env.board_size - 1) or (
                                player == -1 and action[2] == 0) else 0.0

                    current_agent.remember(state, action, reward, next_state, done)

                    if len(current_agent.memory) > current_agent.batch_size:
                        loss = current_agent.replay()
                        if loss is not None:
                            total_loss += loss
                        if current_agent.epsilon > current_agent.epsilon_min:
                            current_agent.epsilon = max(current_agent.epsilon_min,
                                                        current_agent.epsilon * current_agent.epsilon_decay)

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
                winner = self.env.game_winner(state)
                wins[winner] += 1

                win_rate = (wins[1] / max(1, sum(wins.values()))) * 100
                self.win_rates.append(win_rate)
                print(
                    f"Episode {episode + 1}: Win rate {win_rate:.2f}%, Total Wins: {wins[1]}, Losses: {wins[-1]}, Draws: {wins[0]}, Epsilon: {self.epsilons[-1]}")

                if (episode + 1) % self.eval_frequency == 0:
                    eval_score = self.evaluate()
                    self.eval_results.append(eval_score)

        self.save_model(os.path.join(checkpoint_dir, f'model_final_e{self.episodes}.pth'))
        self.plot_training_results()
        return self.agent1, self.agent2, self.rewards, self.win_rates, self.eval_results

    def evaluate(self):
        """ Evaluates the current agent's performance """
        total_wins = 0
        for _ in range(self.eval_games):
            state = self.env.reset()
            done = False
            player = 1
            while not done:
                agent = self.agent1 if player == 1 else self.agent2
                valid_moves = self.env.valid_moves(player)
                if not valid_moves:
                    break
                action = agent.act(state, valid_moves)
                state, _, _, done = self.env.step(action, player)
                player *= -1
            if self.env.game_winner(state) == 1:
                total_wins += 1
        win_rate = (total_wins / self.eval_games) * 100
        print(f"Evaluation: {win_rate:.2f}% win rate over {self.eval_games} games")
        return win_rate

    def plot_training_results(self):
        """ Plot and save training results """
        plt.figure(figsize=(15, 5))

        # Win Rate
        plt.subplot(1, 3, 1)
        plt.plot(range(len(self.win_rates)), self.win_rates, label='Win Rate', color='blue')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate (%)')
        plt.title('Win Rate Progression')
        plt.legend()

        # Rewards
        plt.subplot(1, 3, 2)
        plt.plot(range(len(self.rewards)), self.rewards, label='Rewards per Episode', color='green', alpha=0.7)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('Training Rewards')
        plt.legend()

        # Epsilon Decay
        plt.subplot(1, 3, 3)
        plt.plot(range(len(self.epsilons)), self.epsilons, label='Epsilon Decay', color='red', alpha=0.7)
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay Progression')
        plt.legend()

        plt.tight_layout()
        plt.savefig('checkpoints/training_results.png')
        plt.show()

    def save_model(self, path="checkpoints/best_model.pth"):
        torch.save({
            "model_state_dict": self.agent1.q_network.state_dict(),
            "optimizer_state_dict": self.agent1.optimizer.state_dict(),
            "epsilon": self.agent1.epsilon,
            "win_rate": self.win_rates[-1] if self.win_rates else 0
        }, path)
        print(f"Model saved at {path}")

def load_trained_model(model_path):
    """Load a trained model with correct state_dict keys."""
    agent = DQNAgent()  # Ensure we create an agent with the correct architecture

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=agent.device)

            if "model_state_dict" in checkpoint:
                agent.q_network.load_state_dict(checkpoint["model_state_dict"], strict=False)
                agent.target_network.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                print("Error: Model state_dict not found in checkpoint.")

            if "optimizer_state_dict" in checkpoint:
                agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            agent.epsilon = checkpoint.get("epsilon", agent.epsilon)
            print(f"Loaded model from {model_path} successfully!")
            print(f"Win rate at save: {checkpoint.get('win_rate', 'Unknown'):.2f}%")
        except Exception as e:
            print(f"Error loading {model_path}: {e}")
    else:
        print(f"No trained model found. Using untrained agent.")

    return agent

def train_agent():
    env = checkers_env()
    agent1, agent2 = DQNAgent(), DQNAgent()
    trainer = CheckersTrainer(env, agent1, agent2, parse_args())
    return trainer.train()


if __name__ == "__main__":
    train_agent()
