import torch
import numpy as np
from replay_buffer import ReplayBuffer
from q_network import QNetwork
import gymnasium as gym

class StepWrapper(gym.Wrapper):
    def step(self, action):
        next_state, reward, terminated, truncated, info = self.env.step(action)
        reward += abs(next_state[1])
        if terminated:
            print("finished!")
            reward += 100
        return next_state, reward, terminated, truncated, info

def train(episodes, batch_size, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, buffer_capacity, max_steps):
    env = gym.make('MountainCar-v0', max_episode_steps=max_steps)
    env = StepWrapper(env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    replay_buffer = ReplayBuffer(buffer_capacity)
    epsilon = epsilon

    for episode in range(episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state)
        total_reward = 0

        while True:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(action_dim)
            else:
                q_values = q_net(state.unsqueeze(0))
                action = torch.argmax(q_values).item()

            # Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = torch.FloatTensor(next_state)

            # Add experience to the replay buffer
            replay_buffer.add(state.numpy(), action, reward, next_state.numpy(), terminated)

            # Train if enough samples
            if replay_buffer.size() >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q_values = q_net(next_states).max(1)[0]
                    targets = rewards + gamma * max_next_q_values * (1 - dones)

                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")

    torch.save(q_net.state_dict(), "trained_q_network.pth")
    print("Training complete. Model saved.")
