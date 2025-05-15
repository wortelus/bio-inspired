
import torch
import gymnasium as gym

from n_05.agent import Agent

def visualize_agent(agent: Agent, env_name="CartPole-v1"):
    env_vis = gym.make(env_name, render_mode="human")
    for i in range(5):  # Přehrání 5 epizod
        state, info = env_vis.reset()

        done = False
        total_reward_vis = 0

        print(f"vizualizační epizoda {i + 1}")
        while not done:
            env_vis.render()
            original_epsilon = agent.epsilon
            agent.epsilon = 0.00

            action = agent.select_action(state)

            agent.epsilon = original_epsilon

            next_state, reward, terminated, truncated, _ = env_vis.step(action)
            done = terminated or truncated
            state = next_state
            total_reward_vis += reward
        print(f"celková odměna za epizodu {i + 1}: {total_reward_vis}")

    env_vis.close()

def main():
    # zjištění pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"pytorch zařízení: {device}")

    # tvorba agenta
    agent = Agent(str(device))
    agent.load("dqn_cartpole.pth")
    visualize_agent(agent)

if __name__ == "__main__":
    main()