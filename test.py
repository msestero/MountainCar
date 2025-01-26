import torch
from q_network import QNetwork
import gymnasium as gym

def test(max_steps):
    env = gym.make('MountainCar-v0', max_episode_steps=max_steps, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_net = QNetwork(state_dim, action_dim)
    q_net.load_state_dict(torch.load("trained_q_network.pth"))
    q_net.eval()

    state, _ = env.reset()
    state = torch.FloatTensor(state)

    total_reward = 0
    while True:
        q_values = q_net(state.unsqueeze(0))
        action = torch.argmax(q_values).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = torch.FloatTensor(next_state)
        total_reward += reward

        if terminated or truncated:
            print(f"Test finished with total reward: {total_reward}")
            break

    env.close()
