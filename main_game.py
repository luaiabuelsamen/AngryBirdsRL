import pygame
import math
import random


pygame.init()


WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Angry Birds with Multiple Levels")


SKY_BLUE = (135, 206, 235)
BIRD_RED = (255, 50, 50)
BIRD_YELLOW = (255, 255, 0)
BIRD_BLUE = (50, 50, 255)
GRASS_GREEN = (0, 200, 0)
BLOCK_BROWN = (139, 69, 19)
BOMB_BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


clock = pygame.time.Clock()
gravity = 0.5
bird_radius = 15
bird_pos = [150, 300]
bird_velocity = [0, 0]
bird_moving = False
bird_type = "normal"  
bird_color = BIRD_RED


slingshot_pos = [150, 300]
is_pulled = False
attempts = 0
score = 0
max_attempts = 3
current_level = 0
levels = []  
bombs = []  
game_over_flag = False  


def create_level(blocks_count, block_width, block_height):
    blocks = []
    start_x = 600
    start_y = 450
    for i in range(blocks_count):
        x = start_x + (i % 3) * block_width  
        y = start_y - (i // 3) * block_height  
        blocks.append({"pos": [x, y], "width": block_width, "height": block_height, "color": BLOCK_BROWN, "destroyed": False})
    return blocks


def create_bombs(num_bombs):
    return [
        {"pos": [random.randint(500, 700), random.randint(400, 500)], "radius": 20, "destroyed": False}
        for _ in range(num_bombs)
    ]


for level in range(5):  
    blocks_count = (level + 1) * 3  
    block_width = 50 - level * 5  
    block_height = 50
    levels.append(create_level(blocks_count, block_width, block_height))
    bombs.append(create_bombs(level + 1))  

def draw_bird(pos):
    pygame.draw.circle(screen, bird_color, (int(pos[0]), int(pos[1])), bird_radius)

def draw_blocks(blocks):
    for block in blocks:
        if not block["destroyed"]:
            pygame.draw.rect(screen, block["color"], pygame.Rect(block["pos"][0], block["pos"][1], block["width"], block["height"]))

def draw_bombs(bombs):
    for bomb in bombs:
        if not bomb["destroyed"]:
            pygame.draw.circle(screen, BOMB_BLACK, bomb["pos"], bomb["radius"])

def check_collision(bird_pos, block):
    bird_rect = pygame.Rect(bird_pos[0] - bird_radius, bird_pos[1] - bird_radius, bird_radius * 2, bird_radius * 2)
    block_rect = pygame.Rect(block["pos"][0], block["pos"][1], block["width"], block["height"])
    return bird_rect.colliderect(block_rect)

def check_bomb_collision(bird_pos, bomb):
    distance = math.sqrt((bomb["pos"][0] - bird_pos[0]) ** 2 + (bomb["pos"][1] - bird_pos[1]) ** 2)
    return distance < bomb["radius"] + bird_radius

def reset_bird():
    global bird_pos, bird_velocity, bird_moving, is_pulled, bird_type, bird_color
    bird_pos = [slingshot_pos[0], slingshot_pos[1]]
    bird_velocity = [0, 0]
    bird_moving = False
    is_pulled = False
    
    if bird_type == "normal":
        bird_type = "fast"
        bird_color = BIRD_YELLOW
    elif bird_type == "fast":
        bird_type = "glide"
        bird_color = BIRD_BLUE
    else:
        bird_type = "normal"
        bird_color = BIRD_RED

def draw_trajectory(start, end):
    dx = start[0] - end[0]
    dy = start[1] - end[1]
    for i in range(1, 30, 2):
        dot_x = start[0] - (dx * i / 30)
        dot_y = start[1] - (dy * i / 30) + (0.5 * gravity * (i ** 2) / 10)
        pygame.draw.circle(screen, BLACK, (int(dot_x), int(dot_y)), 2)

def game_over(blocks):
    return all(block["destroyed"] for block in blocks)

def level_failed():
    return attempts >= max_attempts

def reset_level():
    global attempts, bird_moving, bird_pos, levels, bombs, current_level
    attempts = 0
    bird_moving = False
    bird_pos = [slingshot_pos[0], slingshot_pos[1]]

def next_level():
    global current_level, game_over_flag
    current_level += 1
    if current_level >= len(levels):
        game_over_flag = True  

def draw_background():
    
    screen.fill(SKY_BLUE)
    
    
    pygame.draw.rect(screen, GRASS_GREEN, pygame.Rect(0, HEIGHT - 50, WIDTH, 50))
    
    
    pygame.draw.circle(screen, WHITE, (200, 100), 30)
    pygame.draw.circle(screen, WHITE, (230, 100), 40)
    pygame.draw.circle(screen, WHITE, (260, 100), 30)
    
    pygame.draw.circle(screen, WHITE, (500, 80), 30)
    pygame.draw.circle(screen, WHITE, (530, 80), 40)
    pygame.draw.circle(screen, WHITE, (560, 80), 30)


def update_bird_movement():
    global bird_velocity, bird_pos
    if bird_type == "normal":
        bird_velocity[1] += gravity  
    elif bird_type == "fast":
        
        bird_velocity[1] = 0
        bird_velocity[0] *= 1.1  
    elif bird_type == "glide":
        bird_velocity[1] += gravity * 0.3  

    bird_pos[0] += bird_velocity[0]
    bird_pos[1] += bird_velocity[1]


running = True
while running:
    draw_background()

    
    if game_over_flag:
        font = pygame.font.SysFont(None, 55)
        win_text = font.render("Congratulations! All Levels Completed!", True, BLACK)
        screen.blit(win_text, (WIDTH // 2 - 250, HEIGHT // 2))
        pygame.display.update()
        pygame.time.delay(5000)  
        running = False  
        continue  

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            if math.hypot(mouse_pos[0] - slingshot_pos[0], mouse_pos[1] - slingshot_pos[1]) < bird_radius:
                is_pulled = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if is_pulled:
                bird_velocity[0] = (slingshot_pos[0] - pygame.mouse.get_pos()[0]) * 0.2
                bird_velocity[1] = (slingshot_pos[1] - pygame.mouse.get_pos()[1]) * 0.2
                bird_moving = True
                is_pulled = False
                attempts += 1

    
    if bird_moving:
        update_bird_movement()

        
        if bird_pos[1] + bird_radius > HEIGHT - 50:  
            bird_velocity[1] *= -0.7
            bird_pos[1] = HEIGHT - 50 - bird_radius

        
        for block in levels[current_level]:
            if not block["destroyed"] and check_collision(bird_pos, block):
                block["destroyed"] = True
                score += 10
                print(f"Score: {score}")

        
        for bomb in bombs[current_level]:
            if not bomb["destroyed"] and check_bomb_collision(bird_pos, bomb):
                bomb["destroyed"] = True
                print("Bomb exploded!")
                
                for block in levels[current_level]:
                    if math.hypot(bomb["pos"][0] - block["pos"][0], bomb["pos"][1] - block["pos"][1]) < 100:
                        block["destroyed"] = True
                        score += 10
                print(f"Score: {score}")

    
    if is_pulled:
        mouse_pos = pygame.mouse.get_pos()
        bird_pos = list(mouse_pos)
        draw_trajectory(slingshot_pos, mouse_pos)

    
    draw_bird(bird_pos)
    draw_blocks(levels[current_level])
    draw_bombs(bombs[current_level])

    
    if game_over(levels[current_level]):
        font = pygame.font.SysFont(None, 55)
        win_text = font.render("Level Complete!", True, BLACK)
        screen.blit(win_text, (WIDTH // 2 - 150, HEIGHT // 2))
        pygame.display.update()
        pygame.time.delay(2000)  
        next_level()
        reset_level()
    elif level_failed():
        font = pygame.font.SysFont(None, 55)
        lose_text = font.render("Level Failed! Resetting...", True, BLACK)
        screen.blit(lose_text, (WIDTH // 2 - 200, HEIGHT // 2))
        pygame.display.update()
        pygame.time.delay(2000)  
        reset_level()

    else:
        font = pygame.font.SysFont(None, 25)
        score_text = font.render(f"Attempts: {attempts}/{max_attempts}  Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

    
    pygame.display.update()
    clock.tick(60)

    
    if bird_pos[0] > WIDTH or bird_pos[1] > HEIGHT:
        reset_bird()


pygame.quit()
