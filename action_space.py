import math

class ActionMapper:
    def __init__(self):
        # Define the number of discrete actions: 8 directions * 3 pull distances + 1 release action
        self.directions = 8
        self.distances = 3
        self.num_actions = self.directions * self.distances + 1  # Including release action

    def get_action(self, action):
        # Map action integer to direction, distance, and release
        if action == self.num_actions - 1:
            return None, None, True  # Release action
        direction = action % self.directions
        distance = action // self.directions
        return direction, distance, False

    def calculate_pull_position(self, direction, distance, slingshot_pos):
        # Convert direction and distance to a specific position for pulling
        angle = 2 * math.pi * direction / self.directions
        pull_strength = 30 * (distance + 1)  # Adjust pull strength as needed
        x = slingshot_pos[0] - int(pull_strength * math.cos(angle))
        y = slingshot_pos[1] - int(pull_strength * math.sin(angle))
        return [x, y]

        # Calculate velocity based on pull direction and strength
        angle = 2 * math.pi * direction / self.directions
        velocity_strength = 10 * (distance + 1)  # Adjust velocity strength
        vx = -velocity_strength * math.cos(angle)
        vy = -velocity_strength * math.sin(angle)
        return [vx, vy]
