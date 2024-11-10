import pygame
import math
import random

class Game:
    def __init__(self, width=800, height=600, max_attempts=3):
        pygame.init()
        self.WIDTH, self.HEIGHT = width, height
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Enhanced Angry Birds with Multiple Levels")

        # Colors and constants
        self.SKY_BLUE = (135, 206, 235)
        self.BIRD_RED = (255, 50, 50)
        self.BIRD_YELLOW = (255, 255, 0)
        self.BIRD_BLUE = (50, 50, 255)
        self.GRASS_GREEN = (0, 200, 0)
        self.BLOCK_BROWN = (139, 69, 19)
        self.BOMB_BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        self.clock = pygame.time.Clock()
        self.gravity = 0.5
        self.bird_radius = 15
        self.slingshot_pos = [150, 300]
        self.max_attempts = max_attempts

        # Game state variables
        self.bird_pos = [150, 300]
        self.bird_velocity = [0, 0]
        self.bird_moving = False
        self.bird_type = "normal"
        self.bird_color = self.BIRD_RED

        self.is_pulled = False
        self.attempts = 0
        self.score = 0
        self.current_level = 0
        self.levels = []
        self.bombs = []
        self.game_over_flag = False

        # Create initial levels and bombs
        self.create_levels(5)

    def create_levels(self, num_levels):
        for level in range(num_levels):
            blocks_count = (level + 1) * 3
            block_width = 50 - level * 5
            block_height = 50
            self.levels.append(self.create_level(blocks_count, block_width, block_height))
            self.bombs.append(self.create_bombs(level + 1))

    def create_level(self, blocks_count, block_width, block_height):
        blocks = []
        start_x = 600
        start_y = 450
        for i in range(blocks_count):
            x = start_x + (i % 3) * block_width
            y = start_y - (i // 3) * block_height
            blocks.append({"pos": [x, y], "width": block_width, "height": block_height, "color": self.BLOCK_BROWN, "destroyed": False})
        return blocks

    def create_bombs(self, num_bombs):
        return [{"pos": [random.randint(500, 700), random.randint(400, 500)], "radius": 20, "destroyed": False} for _ in range(num_bombs)]

    def reset_bird(self):
        self.bird_pos = [self.slingshot_pos[0], self.slingshot_pos[1]]
        self.bird_velocity = [0, 0]
        self.bird_moving = False
        self.is_pulled = False

        # Rotate bird types for next launch
        if self.bird_type == "normal":
            self.bird_type = "fast"
            self.bird_color = self.BIRD_YELLOW
        elif self.bird_type == "fast":
            self.bird_type = "glide"
            self.bird_color = self.BIRD_BLUE
        else:
            self.bird_type = "normal"
            self.bird_color = self.BIRD_RED

    def draw_background(self):
        self.screen.fill(self.SKY_BLUE)
        pygame.draw.rect(self.screen, self.GRASS_GREEN, pygame.Rect(0, self.HEIGHT - 50, self.WIDTH, 50))

        # Draw clouds
        pygame.draw.circle(self.screen, self.WHITE, (200, 100), 30)
        pygame.draw.circle(self.screen, self.WHITE, (230, 100), 40)
        pygame.draw.circle(self.screen, self.WHITE, (260, 100), 30)
        pygame.draw.circle(self.screen, self.WHITE, (500, 80), 30)
        pygame.draw.circle(self.screen, self.WHITE, (530, 80), 40)
        pygame.draw.circle(self.screen, self.WHITE, (560, 80), 30)

    def update_bird_movement(self):
        if self.bird_type == "normal":
            self.bird_velocity[1] += self.gravity
        elif self.bird_type == "fast":
            self.bird_velocity[1] = 0
            self.bird_velocity[0] *= 1.1
        elif self.bird_type == "glide":
            self.bird_velocity[1] += self.gravity * 0.3

        self.bird_pos[0] += self.bird_velocity[0]
        self.bird_pos[1] += self.bird_velocity[1]

    def draw_game_elements(self):
        self.draw_bird(self.bird_pos)
        self.draw_blocks(self.levels[self.current_level])
        self.draw_bombs(self.bombs[self.current_level])

    def draw_bird(self, pos):
        pygame.draw.circle(self.screen, self.bird_color, (int(pos[0]), int(pos[1])), self.bird_radius)

    def draw_blocks(self, blocks):
        for block in blocks:
            if not block["destroyed"]:
                pygame.draw.rect(self.screen, block["color"], pygame.Rect(block["pos"][0], block["pos"][1], block["width"], block["height"]))
    
    def draw_trajectory(self, start, end):
        dx = start[0] - end[0]
        dy = start[1] - end[1]
        for i in range(1, 30, 2):
            dot_x = start[0] - (dx * i / 30)
            dot_y = start[1] - (dy * i / 30) + (0.5 * self.gravity * (i ** 2) / 10)
            pygame.draw.circle(self.screen, self.BLACK, (int(dot_x), int(dot_y)), 2)

    def draw_bombs(self, bombs):
        for bomb in bombs:
            if not bomb["destroyed"]:
                pygame.draw.circle(self.screen, self.BOMB_BLACK, bomb["pos"], bomb["radius"])

    def check_collision(self, bird_pos, block):
        bird_rect = pygame.Rect(bird_pos[0] - self.bird_radius, bird_pos[1] - self.bird_radius, self.bird_radius * 2, self.bird_radius * 2)
        block_rect = pygame.Rect(block["pos"][0], block["pos"][1], block["width"], block["height"])
        return bird_rect.colliderect(block_rect)

    def check_bomb_collision(self, bird_pos, bomb):
        distance = math.sqrt((bomb["pos"][0] - bird_pos[0]) ** 2 + (bomb["pos"][1] - bird_pos[1]) ** 2)
        return distance < bomb["radius"] + self.bird_radius

    def game_over(self):
        return all(block["destroyed"] for block in self.levels[self.current_level])

    def level_failed(self):
        return self.attempts >= self.max_attempts

    def reset_level(self):
        self.attempts = 0
        self.bird_moving = False
        self.bird_pos = [self.slingshot_pos[0], self.slingshot_pos[1]]
        for block in self.levels[self.current_level]:
            block["destroyed"] = False
        for bomb in self.bombs[self.current_level]:
            bomb["destroyed"] = False

    def next_level(self):
        self.current_level += 1
        if self.current_level >= len(self.levels):
            self.game_over_flag = True

if __name__ == "__main__":
    game = Game()  # Initialize the game

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if math.hypot(mouse_pos[0] - game.slingshot_pos[0], mouse_pos[1] - game.slingshot_pos[1]) < game.bird_radius:
                    game.is_pulled = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if game.is_pulled:
                    # Set bird velocity based on slingshot pull
                    mouse_pos = pygame.mouse.get_pos()
                    game.bird_velocity[0] = (game.slingshot_pos[0] - mouse_pos[0]) * 0.2
                    game.bird_velocity[1] = (game.slingshot_pos[1] - mouse_pos[1]) * 0.2
                    game.bird_moving = True
                    game.is_pulled = False
                    game.attempts += 1

        # Update background
        game.screen.fill(game.SKY_BLUE)
        game.draw_background()

        # Check if bird is being dragged
        if game.is_pulled:
            # Update bird position to follow the mouse while dragging
            mouse_pos = pygame.mouse.get_pos()
            game.bird_pos = list(mouse_pos)
            # Draw the trajectory
            game.draw_trajectory(game.slingshot_pos, mouse_pos)
        elif game.bird_moving:
            # Update bird position if it's moving
            game.update_bird_movement()

            # Check ground collision
            if game.bird_pos[1] + game.bird_radius > game.HEIGHT - 50:
                game.bird_velocity[1] *= -0.7
                game.bird_pos[1] = game.HEIGHT - 50 - game.bird_radius

            # Check collisions with blocks
            for block in game.levels[game.current_level]:
                if not block["destroyed"] and game.check_collision(game.bird_pos, block):
                    block["destroyed"] = True
                    game.score += 10

            # Check collisions with bombs
            for bomb in game.bombs[game.current_level]:
                if not bomb["destroyed"] and game.check_bomb_collision(game.bird_pos, bomb):
                    bomb["destroyed"] = True
                    # Destroy nearby blocks
                    for block in game.levels[game.current_level]:
                        if math.hypot(bomb["pos"][0] - block["pos"][0], bomb["pos"][1] - block["pos"][1]) < 100:
                            block["destroyed"] = True
                            game.score += 10

        # Draw elements
        game.draw_game_elements()

        # Check for level completion or failure
        if game.game_over():
            font = pygame.font.SysFont(None, 55)
            win_text = font.render("Level Complete!", True, game.BLACK)
            game.screen.blit(win_text, (game.WIDTH // 2 - 150, game.HEIGHT // 2))
            pygame.display.update()
            pygame.time.delay(2000)
            game.next_level()
            game.reset_level()
        elif game.level_failed():
            font = pygame.font.SysFont(None, 55)
            lose_text = font.render("Level Failed! Resetting...", True, game.BLACK)
            game.screen.blit(lose_text, (game.WIDTH // 2 - 200, game.HEIGHT // 2))
            pygame.display.update()
            pygame.time.delay(2000)
            game.reset_level()

        # Draw attempts and score
        font = pygame.font.SysFont(None, 25)
        score_text = font.render(f"Attempts: {game.attempts}/{game.max_attempts}  Score: {game.score}", True, game.BLACK)
        game.screen.blit(score_text, (10, 10))

        # Update display
        pygame.display.update()
        game.clock.tick(60)

        # Reset bird if it goes off screen
        if game.bird_pos[0] > game.WIDTH or game.bird_pos[1] > game.HEIGHT:
            game.reset_bird()

    pygame.quit()
