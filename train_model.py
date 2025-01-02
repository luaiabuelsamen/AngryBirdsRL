import os
from datetime import datetime
import random
from collections import deque
import json

import pygame
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from main_game import CartPoleGame

# DQN hyperparameters
MEMORY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
TARGET_UPDATE = 10

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.stack(states)
        next_states = np.stack(next_states)
        return (
            torch.from_numpy(states).float(),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float),
            torch.from_numpy(next_states).float(),
            torch.tensor(dones, dtype=torch.float)
        )
    
    
    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.epsilon = EPSILON_START
        self.steps = 0
        self.start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_logs = []
    
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)
    
    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
        
        loss = nn.MSELoss()(current_q_values, target_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

        self.steps += 1

        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, episode, total_reward):
        timestamp = self.start_time
        model_path = f"models/dqn_model_{timestamp}.pth"
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'steps': self.steps
        }, model_path)
        
        self.training_logs.append({
            'episode': episode,
            'total_reward': total_reward,
            'epsilon': self.epsilon,
            'steps': self.steps
        })
        
        log_path = f"logs/training_log_{timestamp}.json"
        with open(log_path, 'w') as f:
            json.dump(self.training_logs, f, indent=4)
        
        print(f"Saved model to {model_path}")
        print(f"Saved logs to {log_path}")

    def load(self, model_path):
        checkpoint = torch.load(model_path, weights_only = True)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Loaded model from {model_path}")

def train_dqn(episodes=1000):
    env = CartPoleGame(render_mode="human")
    agent = DQNAgent(state_size=4, action_size=2)
    
    try:
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            while True:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                
                agent.memory.push(state, action, reward, next_state, done)
                loss = agent.train()
                
                total_reward += reward
                state = next_state
                
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        agent.save(episode, total_reward)
                        pygame.quit()
                        return
                
                if env.trial_number % 50 == 0:
                    agent.save(episode, total_reward)
                if done:
                    print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")
                    agent.save(episode, total_reward)
                    break
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        agent.save(episode, total_reward)
        pygame.quit()

if __name__ == "__main__":
    episodes = int(input("Enter number of episodes to train (default 1000): ") or 1000)
    train_dqn(episodes)
