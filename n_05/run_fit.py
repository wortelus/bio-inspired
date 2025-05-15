import random

import numpy as np
import torch
from matplotlib import pyplot as plt

from n_05.agent import Agent
from n_05.run_play import visualize_agent


def plot_rewards(episode_rewards, moving_average_rewards, env_name="CartPole-v1"):
    plt.figure(figsize=(12, 6))
    # plt.plot(episode_rewards, label='odměna za epizodu')
    plt.plot(moving_average_rewards,
             label='klouzavý průměr (100 epizod)', color='red')
    plt.xlabel('epizoda')
    plt.ylabel('celková odměna')
    plt.title(f'Trénink DQN agenta pro {env_name}')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_agent():
    SEED = 10
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # zjištění pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch zařízení: {device}")

    # tvorba agenta
    agent = Agent(str(device),
                  epsilon=1.0,
                  epsilon_decay=0.995,
                  epsilon_min=0.01,
                  target_update_every_steps=100,
                  memory_size=10000)

    print("start tréninku...")
    episode_rewards, moving_average_rewards = agent.fit(
        num_episodes=2000,
        # dle dokumentace po 500 epizodách truncate pro CartPole
        max_steps_per_episode=500,
        target_reward=475,
        batch_size=128,
        print_every_episodes=100,
        start_training_after_mem_size=1000
    )
    print("trénink hotov")
    plot_rewards(episode_rewards, moving_average_rewards)
    visualize_agent(agent)


if __name__ == "__main__":
    train_agent()
