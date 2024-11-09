import pygame
import gym
from gym import spaces
import numpy as np
import random
import math

# Initialize pygame
pygame.init()

# Define your game environment as a subclass of gym.Env
class AngryBirdsEnv(gym.Env):
    def __init__(self):
        super(AngryBirdsEnv, self).__init__()

        # Define the action space: Let's use 5 angles and 5 strengths
        self.action_space = spaces.Discrete(25)  # 5 angles * 5 strengths

        # Observation space: bird's position (x, y) and block positions
        self.observation_space = spaces.Box(low=0, high=800, shape=(12,), dtype=np.float32)

        # Initialize game variables
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Angry Birds")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the environment (start a new game)."""
        self.bird_pos = [150, 300]
        self.bird_velocity = [0, 0]
        self.bird_moving = False
        self.blocks = self.create_blocks()  # Assume this function creates blocks for the level
        self.attempts = 0
        self.max_attempts = 3
        return self._get_obs()

    def _get_obs(self):
        """Return the current state as the observation."""
        obs = np.array(self.bird_pos + [block["pos"][0] for block in self.blocks])
        return obs

    def step(self, action):
        """Apply action and return new state, reward, done, info."""
        angle_idx = action // 5
        strength_idx = action % 5
        angle = angle_idx * (math.pi / 4)  # Convert to radians
        strength = (strength_idx + 1) * 10  # Scale strength

        # Apply the action (launch the bird)
        self.bird_velocity = [strength * math.cos(angle), -strength * math.sin(angle)]
        self.bird_moving = True
        self.attempts += 1  # Increment attempts

        # Move bird and check collisions
        self._move_bird()

        # Calculate reward based on collision with blocks or bombs
        reward = self._calculate_reward()

        # Check if the game is done (level completed or attempts over)
        done = self._check_done()

        # Return state, reward, done, and optional info
        return self._get_obs(), reward, done, {}

    def _move_bird(self):
        """Move the bird and handle collisions."""
        if self.bird_moving:
            self.bird_pos[0] += self.bird_velocity[0]
            self.bird_pos[1] += self.bird_velocity[1]
            self.bird_velocity[1] += 0.5  # Gravity

            # Check ground collision
            if self.bird_pos[1] >= 550:  # Hit ground
                self.bird_pos[1] = 550
                self.bird_velocity = [0, 0]
                self.bird_moving = False

    def _calculate_reward(self):
        """Calculate reward based on collisions."""
        reward = 0
        for block in self.blocks:
            if not block["destroyed"] and self._check_collision(block):
                block["destroyed"] = True
                reward += 10  # Give reward for destroying a block
        return reward

    def _check_collision(self, block):
        """Check if the bird collides with a block."""
        return math.hypot(block["pos"][0] - self.bird_pos[0], block["pos"][1] - self.bird_pos[1]) < 20

    def _check_done(self):
        """Check if the game is done (all blocks destroyed or attempts over)."""
        if all(block["destroyed"] for block in self.blocks):
            return True  # Level complete
        if self.attempts >= self.max_attempts:
            return True  # Game over
        return False

    def render(self, mode='human'):
        """Render the game for visualization (using pygame)."""
        self.screen.fill((255, 255, 255))

        # Draw bird
        pygame.draw.circle(self.screen, (255, 0, 0), (int(self.bird_pos[0]), int(self.bird_pos[1])), 10)

        # Draw blocks
        for block in self.blocks:
            if not block["destroyed"]:
                pygame.draw.rect(self.screen, (0, 128, 0), pygame.Rect(block["pos"][0], block["pos"][1], 30, 30))

        pygame.display.flip()
        self.clock.tick(30)

    def create_blocks(self):
        """Create blocks for the level."""
        return [{"pos": [600 + i * 50, 450], "destroyed": False} for i in range(3)]

def play_game():
    env = AngryBirdsEnv()
    done = False
    action = 0  # Default action index
    while not done:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                # Control actions with keyboard
                if event.key == pygame.K_UP:
                    action = (action + 5) % 25
                elif event.key == pygame.K_DOWN:
                    action = (action - 5) % 25
                elif event.key == pygame.K_SPACE:
                    _, _, done, _ = env.step(action)
                    if done:
                        print("Game Over!")
                        env.reset()
                        done = False

        pygame.display.flip()

    pygame.quit()

# Run the game
play_game()
