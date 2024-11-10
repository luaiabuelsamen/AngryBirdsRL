import pygame
import time
from stable_baselines3 import PPO
import os
from game_env import AngryBirdsEnv
from main_game import Game

class AgentVisualizer:
    def __init__(self, env, model_path, save_video=False):
        self.env = env
        self.model = PPO.load(model_path)
        self.save_video = save_video
        if save_video:
            self.frames = []
    
    def capture_frame(self):
        """Capture the current pygame surface as a frame"""
        if self.save_video:
            pygame_surface = pygame.display.get_surface()
            frame = pygame.surfarray.array3d(pygame_surface)
            self.frames.append(frame)
    
    def save_video_frames(self, filename="gameplay.mp4"):
        """Save captured frames as a video"""
        if self.save_video and self.frames:
            import cv2
            import numpy as np
            
            # Convert frames to video
            height, width = self.frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
            
            for frame in self.frames:
                # Convert from RGB to BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame)
            out.release()
    
    def visualize_episode(self, delay=0.03):
        """Run and visualize a single episode"""
        obs = self.env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Take step in environment
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            
            # Render the game
            self.env.render()
            self.capture_frame()
            
            # Add delay for better visualization
            time.sleep(delay)
            
            # Print real-time statistics
            print(f"\rStep: {steps} | Score: {info['score']} | Reward: {total_reward:.2f}", end="")
            
        print(f"\nEpisode finished after {steps} steps with total reward: {total_reward:.2f}")
        return total_reward, steps

def main():
    # Initialize game and environment
    game = Game()
    env = AngryBirdsEnv(game)
    
    # Load and visualize trained model
    visualizer = AgentVisualizer(env, "trained_model/angry_birds_model", save_video=True)
    
    # Run multiple episodes
    num_episodes = 5
    total_rewards = []
    
    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        reward, steps = visualizer.visualize_episode()
        total_rewards.append(reward)
    
    # Print summary statistics
    print("\nVisualization Summary:")
    print(f"Average Reward: {sum(total_rewards) / len(total_rewards):.2f}")
    print(f"Best Episode Reward: {max(total_rewards):.2f}")
    print(f"Worst Episode Reward: {min(total_rewards):.2f}")
    
    # Save video if enabled
    visualizer.save_video_frames()

if __name__ == "__main__":
    main()