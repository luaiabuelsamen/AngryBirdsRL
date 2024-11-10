import gym
from gym import spaces
import numpy as np
import math
from enum import Enum

class BirdAction(Enum):
    PULL = 0
    RELEASE = 1

class AngryBirdsEnv(gym.Env):
    def __init__(self, game):
        super(AngryBirdsEnv, self).__init__()
        self.game = game
        
        # Simplified action space with continuous values
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, 0.0]),  # angle, power, release
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # Enhanced observation space with normalized values
        self.observation_space = spaces.Dict({
            'bird_state': spaces.Box(
                low=np.array([0, 0, -1, -1, 0]),  # x, y, vx, vy, bird_type
                high=np.array([1, 1, 1, 1, 2]),
                dtype=np.float32
            ),
            'target_state': spaces.Box(
                low=np.array([0, 0, 0, 0]),  # closest_x, closest_y, distance, num_blocks
                high=np.array([1, 1, 1, 20]),
                dtype=np.float32
            ),
            'game_state': spaces.Box(
                low=np.array([0, 0, 0]),  # attempts, score, level_progress
                high=np.array([3, 1000, 1]),
                dtype=np.float32
            )
        })
        
        self.previous_score = 0
        self.previous_blocks = 0
        self.steps_since_launch = 0
        self.max_steps_per_episode = 300
        self.current_step = 0
        self.episode_reward = 0
        
        # Performance tracking
        self.success_rate = []
        self.episode_lengths = []
        self.episode_rewards = []

    def get_normalized_observation(self):
        """Get normalized observation state"""
        # Bird state
        bird_pos_x = self.game.bird_pos[0] / self.game.WIDTH
        bird_pos_y = self.game.bird_pos[1] / self.game.HEIGHT
        bird_vel_x = np.clip(self.game.bird_velocity[0] / 50.0, -1, 1)
        bird_vel_y = np.clip(self.game.bird_velocity[1] / 50.0, -1, 1)
        bird_type_map = {"normal": 0, "fast": 1, "glide": 2}
        
        bird_state = np.array([
            bird_pos_x, bird_pos_y, bird_vel_x, bird_vel_y,
            bird_type_map[self.game.bird_type]
        ], dtype=np.float32)

        # Target state
        active_blocks = [b for b in self.game.levels[self.game.current_level] 
                        if not b["destroyed"]]
        if active_blocks:
            distances = [math.hypot(b["pos"][0] - self.game.bird_pos[0],
                                  b["pos"][1] - self.game.bird_pos[1])
                        for b in active_blocks]
            closest_idx = np.argmin(distances)
            closest_block = active_blocks[closest_idx]
            
            target_state = np.array([
                closest_block["pos"][0] / self.game.WIDTH,
                closest_block["pos"][1] / self.game.HEIGHT,
                min(1.0, min(distances) / self.game.WIDTH),  # Normalized distance
                len(active_blocks) / 20.0  # Normalize block count
            ], dtype=np.float32)
        else:
            target_state = np.zeros(4, dtype=np.float32)

        # Game state
        game_state = np.array([
            self.game.attempts / self.game.max_attempts,
            self.game.score / 1000.0,  # Normalize score
            len([b for b in self.game.levels[self.game.current_level] 
                 if b["destroyed"]]) / len(self.game.levels[self.game.current_level])
        ], dtype=np.float32)

        return {
            'bird_state': bird_state,
            'target_state': target_state,
            'game_state': game_state
        }

    def calculate_reward(self):
        """Calculate reward with multiple components"""
        reward = 0
        
        # Score-based reward
        score_diff = self.game.score - self.previous_score
        reward += score_diff * 0.5
        
        # Block destruction reward
        current_blocks = len([b for b in self.game.levels[self.game.current_level] 
                            if not b["destroyed"]])
        blocks_destroyed = self.previous_blocks - current_blocks
        reward += blocks_destroyed * 5
        
        # Distance-based reward
        active_blocks = [b for b in self.game.levels[self.game.current_level] 
                        if not b["destroyed"]]
        if active_blocks and self.game.bird_moving:
            min_distance = min(math.hypot(b["pos"][0] - self.game.bird_pos[0],
                                        b["pos"][1] - self.game.bird_pos[1])
                             for b in active_blocks)
            reward += 2.0 / (1.0 + min_distance / 100.0)
        
        # Completion rewards/penalties
        if self.game.game_over():
            completion_bonus = 100 * (1 - self.steps_since_launch / self.max_steps_per_episode)
            reward += completion_bonus
        elif self.game.level_failed():
            reward -= 50
        
        # Step penalty to encourage efficiency
        reward -= 0.1
        
        return reward

    def step(self, action):
        """Execute action in environment"""
        self.current_step += 1
        self.steps_since_launch += 1
        
        # Convert continuous actions to game controls
        angle = np.pi * action[0]  # Convert [-1,1] to [-π,π]
        power = (action[1] + 1) * 25  # Convert [-1,1] to [0,50]
        should_release = action[2] > 0.5
        
        # Store previous state
        self.previous_blocks = len([b for b in self.game.levels[self.game.current_level] 
                                  if not b["destroyed"]])
        self.previous_score = self.game.score
        
        # Apply action
        if not self.game.bird_moving and should_release:
            self.game.bird_velocity = [
                -power * math.cos(angle),
                -power * math.sin(angle)
            ]
            self.game.bird_moving = True
            self.steps_since_launch = 0
        elif not should_release:
            pull_pos = [
                self.game.slingshot_pos[0] - power * math.cos(angle),
                self.game.slingshot_pos[1] - power * math.sin(angle)
            ]
            self.game.bird_pos = [
                np.clip(pull_pos[0], 0, self.game.WIDTH),
                np.clip(pull_pos[1], 0, self.game.HEIGHT)
            ]
        
        # Update game state
        if self.game.bird_moving:
            self.game.update_bird_movement()
        
        # Get new observation
        observation = self.get_normalized_observation()
        
        # Calculate reward
        reward = self.calculate_reward()
        self.episode_reward += reward
        
        # Check if episode is done
        done = (self.game.game_over() or 
                self.game.level_failed() or
                self.current_step >= self.max_steps_per_episode or
                not (0 <= self.game.bird_pos[0] <= self.game.WIDTH and
                     0 <= self.game.bird_pos[1] <= self.game.HEIGHT))
        
        # Update episode statistics if done
        if done:
            self.success_rate.append(1.0 if self.game.game_over() else 0.0)
            self.episode_lengths.append(self.current_step)
            self.episode_rewards.append(self.episode_reward)
            
            # Keep only last 100 episodes
            if len(self.success_rate) > 100:
                self.success_rate = self.success_rate[-100:]
                self.episode_lengths = self.episode_lengths[-100:]
                self.episode_rewards = self.episode_rewards[-100:]
        
        info = {
            'score': self.game.score,
            'blocks_remaining': self.previous_blocks,
            'steps_since_launch': self.steps_since_launch
        }
        
        return observation, reward, done, info

    def reset(self):
        """Reset environment to initial state"""
        self.game.reset_level()
        self.previous_score = 0
        self.previous_blocks = len(self.game.levels[self.game.current_level])
        self.steps_since_launch = 0
        self.current_step = 0
        self.episode_reward = 0
        return self.get_normalized_observation()

    def get_statistics(self):
        """Get training statistics"""
        return {
            'success_rate': np.mean(self.success_rate) if self.success_rate else 0.0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0
        }