import time
import torch
import numpy as np
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from sympy.abc import alpha

from checkers_env import checkers_env
from DQNAgent import DQNAgent
import matplotlib.pyplot as plt
import os
import random
import argparse
import math

# Training parameters
EPISODES = 400  # Increased training episodes for better convergence
EVAL_FREQUENCY = 20  # More frequent evaluations
EVAL_EPISODES = 500  # More evaluation games for accuracy
BATCH_SIZE = 128  # Adjusted batch size for stability
TARGET_UPDATE = 40  # More frequent target updates for stability
MEMORY_SIZE = 500000  # Increased memory size for better experience replay
LEARNING_RATE = 0.001  # Adjusted learning rate for better convergence
EPSILON_START = 1.0
EPSILON_END = 0.1  # Higher minimum epsilon to ensure exploration
EPSILON_DECAY = 0.9998  # Adjusted decay for better long-term learning
GAMMA = 0.99
TAU = 0.005  # Slower soft updates for target network stability
GRADIENT_CLIP = 1.0  # Gradient clipping to prevent instability
PRIORITY_EPSILON = 1e-6

def parse_args():
    parser = argparse.ArgumentParser(description='Train Checkers AI')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes to train')
    parser.add_argument('--eval-frequency', type=int, default=EVAL_FREQUENCY, help='Evaluation frequency')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for replay')
    parser.add_argument('--eval-games', type=int, default=EVAL_EPISODES, help='Number of evaluation games')
    return parser.parse_args()


class CheckersTrainer:
    def __init__(self, env, args):
        self.env = env
        agent1, agent2 = DQNAgent(), DQNAgent()
        self.agent1 = agent1
        self.agent2 = agent2
        self.episodes = args.episodes
        self.eval_frequency = args.eval_frequency
        self.eval_games = args.eval_games
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.epsilon = EPSILON_START
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA
        self.epsilon_min = EPSILON_END
        self.rewards = []
        self.win_rates = []
        self.losses = []
        self.epsilons = []
        self.eval_results = []



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
                self.epsilons.append(self.agent1.epsilon)

                player = random.choice([1, -1])
                total_reward = 0
                total_loss = 0
                move_count = 0
                while not done:
                    valid_moves = self.env.valid_moves(player)
                    if not valid_moves:
                        break

                    if player == 1:
                        action = self.agent1.act(state, valid_moves)
                    else:
                        action = random.choice(valid_moves)

                    next_state, reward, additional_moves, done = self.env.step(action, player)

                    # reward = max(-1, min(reward, 1))  # Normalized rewards to prevent overfitting

                    if player == 1:
                        self.agent1.remember(state, action, reward, next_state, done)

                        if len(self.agent1.memory) > self.batch_size:
                            loss = self.agent1.replay()
                            if loss is not None:
                                total_loss += loss

                    state = next_state
                    total_reward += reward
                    move_count += 1

                    if not additional_moves:
                        player *= -1

                if episode % TARGET_UPDATE == 0:
                    self.agent1.update_target_network()
                if episode % 100 == 0:
                    print(f"Episode {episode}: Avg Loss: {total_loss / max(1, move_count):.5f}")

                self.rewards.append(total_reward)
                self.losses.append(total_loss / max(1, move_count))

                winner = self.env.game_winner(state)
                wins[winner] += 1



                win_rate = (wins[1] / max(1, wins[1] + wins[-1])) * 100
                self.win_rates.append(win_rate)
                print(f"Episode {episode + 1}: Win rate {win_rate:.2f}%, Total Wins: {wins[1]}, Losses: {wins[-1]}, Draws: {wins[0]}, Epsilon: {self.epsilons[-1]}")

                if (episode + 1) % self.eval_frequency == 0:
                    eval_score = self.evaluate(opponent="random")
                    self.eval_results.append(eval_score)

        self.save_model(os.path.join(checkpoint_dir, f'model_final_e{self.episodes}.pth'))
        self.plot_training_results()
        return self.agent1, self.agent2, self.rewards, self.win_rates, self.eval_results

    def evaluate(self, opponent="random"):
        """
        Evaluates the DQN agent by playing against different types of opponents.

        Args:
            opponent (str): Type of opponent ("random", "minimax", "dqn").
            episodes (int): Number of evaluation episodes.

        Returns:
            float: Win rate of the agent.
        """
        wins = {1: 0, -1: 0, 0: 0}  # Tracking wins, losses, draws

        for episode in range(self.eval_games):
            state = self.env.reset()
            done = False
            player = 1  # The trained agent always starts

            while not done:
                valid_moves = self.env.valid_moves(player)
                if not valid_moves:
                    break  # No valid moves → switch player

                if player == 1:
                    action = self.agent1.act(state, valid_moves)  # DQN agent move
                else:
                    if opponent == "random":
                        action = random.choice(valid_moves)  # Random moves
                    elif opponent == "minimax":
                        action = self.env.minimax_move(state, player)  # Use minimax strategy
                    elif opponent == "dqn":
                        action = self.agent2.act(state, valid_moves)  # Another trained agent
                    else:
                        raise ValueError("Invalid opponent type: choose 'random', 'minimax', or 'dqn'.")

                next_state, reward, additional_moves, done = self.env.step(action, player)
                state = next_state

                if not additional_moves:
                    player *= -1  # Switch turns

            winner = self.env.game_winner(state)
            wins[winner] += 1

        win_rate = (wins[1] / max(1, wins[1] + wins[-1])) * 100
        print(
            f"Evaluation against {opponent}: Win rate: {win_rate:.2f}% ({wins[1]} wins, {wins[-1]} losses, {wins[0]} draws)")

        return win_rate  # Returns win percentage



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
    trainer = CheckersTrainer(env, parse_args())
    return trainer.train()


if __name__ == "__main__":
    train_agent()
