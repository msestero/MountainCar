import argparse
from train import train
from test import test

def main():
    parser = argparse.ArgumentParser(description="MountainCar Q-Learning")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Choose whether to train or test the model.")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for training.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for Q-learning.")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon for exploration.")
    parser.add_argument("--epsilon_decay", type=float, default=0.995, help="Decay rate for epsilon.")
    parser.add_argument("--min_epsilon", type=float, default=0.01, help="Minimum value of epsilon.")
    parser.add_argument("--buffer_capacity", type=int, default=10000, help="Capacity of the replay buffer.")
    parser.add_argument("--max_steps", type=int, default=200, help="Maximum steps per episode.")
    
    args = parser.parse_args()

    if args.mode == "train":
        train(
            episodes=args.episodes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            min_epsilon=args.min_epsilon,
            buffer_capacity=args.buffer_capacity,
            max_steps=args.max_steps,
        )
    elif args.mode == "test":
        test(max_steps=args.max_steps)

if __name__ == "__main__":
    main()
