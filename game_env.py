import gym
from gym import spaces
import numpy as np
import pygame
from action_space import ActionMapper

class AngryBirdsEnv(gym.Env):
    def __init__(self, game):
        super(AngryBirdsEnv, self).__init__()

        
        self.action_mapper = ActionMapper()  
        self.action_space = spaces.Discrete(self.action_mapper.num_actions)  
        self.observation_space = spaces.Box(low=0, high=800, shape=(10,), dtype=np.float32)  

        
        self.game = game

    def step(self, action):
        
        pull_direction, pull_distance, release = self.action_mapper.get_action(action)

        
        if not release:
            
            pull_pos = self.action_mapper.calculate_pull_position(pull_direction, pull_distance, self.game.slingshot_pos)
            self.game.bird_pos = pull_pos
            self.game.is_pulled = True
        else:
            
            self.game.is_pulled = False
            self.game.bird_velocity = self.action_mapper.calculate_velocity(pull_direction, pull_distance)
            self.game.bird_moving = True

        
        self.game.update_bird_movement()  
        observation = self.get_observation()
        reward = self.calculate_reward()
        done = self.game_over() or self.level_failed()
        info = {}

        return observation, reward, done, info

    def reset(self):
        self.game.reset_level()
        return self.get_observation()

    def render(self, mode='human'):
        self.game.draw_background()
        self.game.draw_bird(self.game.bird_pos)
        self.game.draw_blocks(self.game.levels[self.game.current_level])
        self.game.draw_bombs(self.game.bombs[self.game.current_level])
        pygame.display.update()

    def get_observation(self):
        
        bird_pos_x, bird_pos_y = self.game.bird_pos
        bird_velocity_x, bird_velocity_y = self.game.bird_velocity
        score = self.game.score
        attempts = self.game.attempts
        return np.array([bird_pos_x, bird_pos_y, bird_velocity_x, bird_velocity_y, score, attempts])

    def calculate_reward(self):
        
        return self.game.score

    def game_over(self):
        return self.game.game_over_flag

    def level_failed(self):
        return self.game.level_failed()
