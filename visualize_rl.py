import os

import pygame
from main_game import CartPoleGame
from train_model import DQNAgent

def play_game(model_path=None):
    env = CartPoleGame(render_mode="human")
    agent = DQNAgent(state_size=4, action_size=2)
    
    if model_path:
        agent.load(model_path)
    
    try:
        while True:
            state = env.reset()
            total_reward = 0
            
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                keys = pygame.key.get_pressed()
                action_type = "Human"
                if keys[pygame.K_LEFT]:
                    action = 0
                elif keys[pygame.K_RIGHT]:
                    action = 1
                else:
                    action_type = "Agent"
                    action = agent.select_action(state)
            
                next_state, reward, done = env.step(action, action_type)
                total_reward += reward
                state = next_state
                
                if done:
                    print(f"Game Over! Total Reward: {total_reward}")
                    break
    
    except KeyboardInterrupt:
        pygame.quit()

if __name__ == "__main__":
    model_files = [f for f in os.listdir("models") if f.endswith('.pth')]
    if not model_files:
        print("No pretrain")
    else:
        play_game('models/' + model_files[0])