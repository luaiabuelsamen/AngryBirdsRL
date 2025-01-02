import pygame
import math
import random
import numpy as np

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
CART_WIDTH = 100
CART_HEIGHT = 20
POLE_LENGTH = 150
GRAVITY = 0.1
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

class CartPoleGame:
    def __init__(self, render_mode="human"):
        self.render_mode = render_mode
        if render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("CartPole DQN")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        

        self.running = True
        self.time_elapsed = 0
        self.trial_number = 1

        self.reset()

    def get_state(self):
        return np.array([
            self.cart_x / SCREEN_WIDTH,
            self.cart_vx / 5.0,
            self.pole_angle / (math.pi/2),
            self.pole_angular_velocity / 2.0
        ])

    def reset(self):
        self.cart_x = SCREEN_WIDTH // 2
        self.cart_vx = 0
        self.pole_angle = random.uniform(-math.pi / 12, math.pi / 12)
        self.pole_angular_velocity = 0
        self.time_elapsed = 0
        self.trial_number += 1
        return self.get_state()

    def apply_force(self, force):
        pole_mass = 0.1
        cart_mass = 1.0
        total_mass = cart_mass + pole_mass
        pole_length = POLE_LENGTH / 100
        
        pole_acceleration = (
            GRAVITY * math.sin(self.pole_angle) - 
            math.cos(self.pole_angle) * (
                force + pole_mass * pole_length * self.pole_angular_velocity**2 * math.sin(self.pole_angle)
            ) / total_mass
        ) / (
            pole_length * (4 / 3 - pole_mass * math.cos(self.pole_angle)**2 / total_mass)
        )

        cart_acceleration = (
            force + pole_mass * pole_length * (
                self.pole_angular_velocity**2 * math.sin(self.pole_angle) - pole_acceleration * math.cos(self.pole_angle)
            )
        ) / total_mass

        self.cart_vx += cart_acceleration / FPS
        self.pole_angular_velocity += pole_acceleration / FPS
        self.cart_x += self.cart_vx
        self.pole_angle += self.pole_angular_velocity

        self.cart_vx = max(min(self.cart_vx, 5), -5)
        self.pole_angular_velocity = max(min(self.pole_angular_velocity, 2), -2)

    def step(self, action, move_type=None):
        if move_type:
            print(f"Move: {action}\tType: {move_type}")
        force = 0.2 if action == 1 else -0.2
        self.apply_force(force)
        
        done = self.check_game_over()
        reward = 1.0 if not done else 0.0
        
        self.time_elapsed += 1
        
        if self.render_mode == "human":
            self.render()
            self.clock.tick(FPS)
        
        return self.get_state(), reward, done

    def render(self):
        if self.render_mode != "human":
            return

        self.screen.fill(WHITE)

        # Draw cart
        cart_rect = pygame.Rect(
            self.cart_x - CART_WIDTH // 2,
            SCREEN_HEIGHT - 100,
            CART_WIDTH,
            CART_HEIGHT,
        )
        pygame.draw.rect(self.screen, BLUE, cart_rect)

        # Draw pole
        pole_x = self.cart_x + POLE_LENGTH * math.sin(self.pole_angle)
        pole_y = SCREEN_HEIGHT - 100 - POLE_LENGTH * math.cos(self.pole_angle)
        pygame.draw.line(
            self.screen, RED,
            (self.cart_x, SCREEN_HEIGHT - 100),
            (pole_x, pole_y),
            5
        )

        # Draw text
        time_text = self.font.render(f"Time: {self.time_elapsed // FPS} s", True, BLACK)
        trial_text = self.font.render(f"Trial: {self.trial_number}", True, BLACK)
        self.screen.blit(time_text, (10, 10))
        self.screen.blit(trial_text, (10, 50))

        pygame.display.flip()

    def check_game_over(self):
        if self.cart_x < 0 or self.cart_x > SCREEN_WIDTH or abs(self.pole_angle) > math.pi / 2:
            return True
        return False

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            keys = pygame.key.get_pressed()
            force = 0
            if keys[pygame.K_LEFT]:
                force = -0.2
            if keys[pygame.K_RIGHT]:
                force = 0.2

            self.apply_force(force)

            if self.check_game_over():
                self.reset()

            self.time_elapsed += 1
            self.render()
            self.clock.tick(FPS)

        pygame.quit()

if __name__ == "__main__":
    game = CartPoleGame()
    game.run()
