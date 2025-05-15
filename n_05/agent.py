import random

import numpy as np
import torch
import torch.nn.functional as F

from n_05.cartpole import CartPole
from n_05.memory import Memory, QMemoryItem
from n_05.nn import QNetwork


class Agent:
    def __init__(self,
                 device: str,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01,
                 gamma=0.99,
                 learning_rate=1e-3,
                 target_update_every_steps=1000,
                 memory_size=10000):
        self.env = CartPole()

        dims = self.env.get_dims()
        action_dim = dims["action_dim"]
        state_dim = dims["state_dim"]

        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        # hlavní síť
        self.network = QNetwork(state_dim, action_dim).to(self.device)

        # cílová síť (má zamrzlé váhy, které se periodicky kopírují z hlavní sítě)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        # inicializace cílové sítě na stejné váhy jako hlavní síť
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()

        # perioda aktualizace cílové sítě
        self.train_step_counter = 0
        self.target_update_every_steps = target_update_every_steps

        self.memory = Memory(capacity=memory_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)

    def select_action(self, state):
        if random.random() < self.epsilon:
            # explorace - náhodná akce
            return random.randrange(self.action_dim)
        else:
            # exploitace - akce s nejvyšší predikovanou Q hodnotou
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.network(state_tensor)
                return q_values.argmax().item()

    def store(self, item: QMemoryItem):
        self.memory.push(item)

    def step(self, batch_size):
        if len(self.memory) < batch_size:
            raise ValueError(f"Nedostatek dat v paměti pro trénink ({len(self.memory)} < {batch_size})")

        # sample batch_size vzorků z paměti
        experiences = self.memory.sample(batch_size)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = experiences

        # pytorch tensor na daný device
        batch_states = torch.FloatTensor(batch_states).to(self.device)
        batch_actions = torch.LongTensor(batch_actions).unsqueeze(1).to(self.device)
        batch_rewards = torch.FloatTensor(batch_rewards).unsqueeze(1).to(self.device)
        batch_next_states = torch.FloatTensor(batch_next_states).to(self.device)
        batch_dones = torch.FloatTensor(batch_dones).unsqueeze(1).to(self.device)

        # dostaneme Q hodnoty pro aktuální stavy a provedné akce z hlavní sítě (self.network)
        current_q_values = self.network(batch_states).gather(1, batch_actions)

        # dostaneme Q hodnoty pro následující stavy z cílové sítě (self.target_network)
        with torch.no_grad():
            next_q_values_from_target_net = self.target_network(batch_next_states).max(1)[0].unsqueeze(1)

        # výpočet očekávaných Q hodnot
        expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * next_q_values_from_target_net

        # ztrátová funkce - MSE
        loss = F.mse_loss(current_q_values, expected_q_values)

        # optimalizace hlavní sítě
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodická aktualizace vah cílové sítě
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_every_steps == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            # print(f"step {self.train_step_counter}: target_net aktualizována...")

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def fit(self,
            num_episodes,
            max_steps_per_episode,
            target_reward,
            batch_size,
            start_training_after_mem_size,
            print_every_episodes=10):
        self.train_step_counter = 0
        episode_rewards = []
        moving_average_rewards = []

        for episode in range(1, num_episodes + 1):
            # pro každou epizodu začíná prostředí v defaultním stavu
            state, info = self.env.reset()
            current_episode_reward = 0

            # simulace prostředí
            for step in range(max_steps_per_episode):
                # výběr akce per epsilon explorace/exploitace
                action = self.select_action(state)

                # provedení akce v prostředí
                # next_state - [pozice vozíku, rychlost vozíku, úhel tyče, úhlová rychlost tyče]
                # reward - odměna za akci (1.0 za úspěšné udržení tyče)
                # terminated - True pokud je překročena některé z termination podmínek
                # ref: https://www.gymlibrary.dev/environments/classic_control/cart_pole/#episode-end
                next_state, reward, terminated = self.env.step(action)

                # uložení akce a jejího výsledku do paměti
                item = QMemoryItem(state, action, reward, next_state, terminated)
                self.store(item)

                if len(self.memory) == start_training_after_mem_size:
                    print(f'paměť naplněna na {start_training_after_mem_size}, začínám trénovat')
                if len(self.memory) >= start_training_after_mem_size:
                    #
                    # trénink agenta, pokud je v paměti dostatek vzorků
                    #
                    self.step(batch_size)

                state = next_state
                current_episode_reward += reward

                # pokud nám CartPole spadne, nebo dokončí limit (500) epizody, ukončíme epizodu
                if terminated:
                    break

            # snížíme epsilon po každé epizodě, aby se zvýšila exploitace a snížila explorace
            self.update_epsilon()

            # log poslední odměny
            episode_rewards.append(current_episode_reward)

            # průměr odměny za posledních n epizod
            # + výpis
            n_mean_episodes = 100
            if episode >= n_mean_episodes:
                avg_reward = np.mean(episode_rewards[-n_mean_episodes:])
                moving_average_rewards.append(avg_reward)
                if avg_reward >= target_reward:
                    print(f"\nProstředí vyřešeno v {episode} epizodách!")
                    print(f"Průměrná odměna za posledních {n_mean_episodes} epizod: {avg_reward:.2f}")
                    break
            else:
                moving_average_rewards.append(np.mean(episode_rewards))

            if print_every_episodes != 0 and episode % print_every_episodes == 0:
                print(f"Epizoda: {episode}/{num_episodes} | "
                      f"poslední odměna: {current_episode_reward:.2f} | "
                      f"Prům. odměna za posledních {n_mean_episodes} epizod: {moving_average_rewards[-1]:.2f} | "
                      f"epsilon: {self.epsilon:.2f} | ")

        # ukončení prostředí
        self.env.close()

        save_path = "dqn_cartpole.pth"
        torch.save(self.network.state_dict(), save_path)
        print(f"model uložen jako {save_path}")
        return episode_rewards, moving_average_rewards

    def load(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.eval()
        print(f"Model načten z {path}")
