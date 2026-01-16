"""
Q-Learning agent for solving the LunarLander-v3 environment.

pip:
  pip install gymnasium[box2d, other] numpy
  python main.py
uv:
  uv run main.py

Autorzy: Hubert Jóźwiak, Kacper Olejnik
"""

import sys
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from collections import defaultdict


class QLearningAgent:
    """Q-Learning agent with epsilon-greedy exploration.

    This agent uses a discretized state space and Q-table to learn optimal
    actions for the LunarLander environment.
    """

    def __init__(
        self,
        n_actions,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
    ):
        """Initialize the Q-Learning agent.

        Args:
            n_actions (int): Number of discrete actions available.
            learning_rate (float): Learning rate (alpha) for Q-value updates.
            discount_factor (float): Discount factor (gamma) for future rewards.
            epsilon (float): Initial exploration rate for epsilon-greedy policy.
            epsilon_decay (float): Multiplicative decay factor for epsilon.
            epsilon_min (float): Minimum epsilon value to maintain exploration.
        """
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

        self.bins = [
            np.linspace(-1.5, 1.5, 10),
            np.linspace(-1.5, 1.5, 10),
            np.linspace(-2, 2, 10),
            np.linspace(-2, 2, 10),
            np.linspace(-np.pi, np.pi, 10),
            np.linspace(-2, 2, 10),
            np.array([0, 1]),
            np.array([0, 1]),
        ]

    def discretize_state(self, state):
        """Convert continuous state to discrete state representation.

        Discretizes each dimension of the state space into bins to make the
        state space manageable for tabular Q-learning.

        Args:
            state (np.ndarray): Continuous state from the environment.

        Returns:
            tuple: Discretized state as a tuple of bin indices.
        """
        discrete_state = tuple(
            np.digitize(state[i], self.bins[i]) for i in range(len(state))
        )

        return discrete_state

    def get_action(self, state, training=True):
        """Select an action using epsilon-greedy policy.

        During training, explores with probability epsilon, otherwise exploits
        by choosing the action with highest Q-value. During testing, always
        exploits the learned policy.

        Args:
            state (np.ndarray): Current state from the environment.
            training (bool): Whether the agent is in training mode.

        Returns:
            int: Selected action index.
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        discrete_state = self.discretize_state(state)
        return np.argmax(self.q_table[discrete_state])

    def update(self, state, action, reward, next_state, done):
        """Update Q-value using the Q-learning update rule.

        Applies the temporal difference update: Q(s,a) <- Q(s,a) + lr * (target - Q(s,a))
        where target = reward (if terminal) or reward + gamma * max_a' Q(s',a').

        Args:
            state (np.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after taking action.
            done (bool): Whether the episode terminated.
        """
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        current_q = self.q_table[discrete_state][action]

        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[discrete_next_state])

        self.q_table[discrete_state][action] += self.lr * (target_q - current_q)

    def decay_epsilon(self):
        """Decay exploration rate (epsilon) after each episode.

        Multiplies epsilon by decay factor, ensuring it doesn't fall below
        the minimum threshold.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_agent(n_episodes=1000):
    """Train a Q-learning agent on the LunarLander environment.

    Args:
        n_episodes (int): Number of training episodes to run.

    Returns:
        QLearningAgent: Trained agent with learned Q-table.
    """
    env = gym.make("LunarLander-v3")
    agent = QLearningAgent(n_actions=env.action_space.n)

    recent_rewards = []

    print(f"LR: {agent.lr}, Gamma: {agent.gamma}, Epsilon: {agent.epsilon}\n")

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, training=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        recent_rewards.append(total_reward)
        if len(recent_rewards) > 100:
            recent_rewards.pop(0)

        if (episode + 1) % 10 == 0:
            avg = np.mean(recent_rewards)
            print(
                f"Episode {episode + 1:4d} | Reward: {total_reward:7.2f} | "
                f"Avg: {avg:7.2f} | ε: {agent.epsilon:.3f}"
            )

    env.close()

    return agent


def test_agent(agent, n_episodes=5):
    """Test a trained agent and record videos of its performance.

    Args:
        agent (QLearningAgent): Trained agent to test.
        n_episodes (int): Number of test episodes to run.
        record_video (bool): Whether to record videos of the episodes.
    """
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    env = RecordVideo(
        env,
        video_folder="./nagrania",
        episode_trigger=lambda x: True,
        name_prefix="lunar_lander",
    )

    for episode in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            total_reward += reward

        print(f"Episode {episode + 1} | Reward: {total_reward:.2f}")

    env.close()


def main() -> int:
    n_train_episodes = 10000
    n_test_episodes = 5

    agent = train_agent(n_episodes=n_train_episodes)
    test_agent(agent, n_episodes=n_test_episodes)

    return 0


if __name__ == "__main__":
    sys.exit(main())
