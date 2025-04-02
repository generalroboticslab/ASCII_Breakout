import numpy as np
import random
import gymnasium as gym
import math

class AsciiAsteroids():
    metadata = {"render_modes": ["human"], "render_fps": 20}

    def __init__(self, size=30):
        self.size = size
        self.width = self.size*2
        self.height = self.size
        self.reset(seed=0)
        self.lives = 3
    
    def reset(self, seed=0):
        #random.seed(seed)
        self.ship = {'x': self.width // 2, 'y': self.height // 2, 'angle': 0, 'vx': 0, 'vy': 0}
        self.asteroids = []
        for _ in range(8):
            x = random.randint(5, self.width - 6)
            y = random.randint(5, self.height - 6)
            vx = random.choice([-1, 0, 1])
            if vx == 0:
                vy = random.choice([-1, 1])
            else:
                vy = random.choice([-1, 0, 1])
            shape = random.choice([[" O ", "OOO"," O "], [" OO", "OOO"," O "], ["OO ", "OOO"," O "],
                                  [" O ", "OOO","OO "],[" O ", "OOO"," OO"],
                                  ["OOO", "OOO","OOO"], [" OOO", "OOOO","OOOO", "OOO "], ["OOO ", "OOOO","OOOO", " OOO"],])
            self.asteroids.append({'x': x, 'y': y, 'vx': vx, 'vy': vy, 'shape': shape})
        
        self.bullets = []
        self.done = False
        return self.render(), {}
    
    
    def step(self, action):
        if self.done:
            return self.render(), 0, True
        
        reward = 0
        
        old_x, old_y = self.ship['x'], self.ship['y']
        
        if action == 1:  # Rotate left
            self.ship['angle'] = (self.ship['angle'] - 90) % 360
        elif action == 2:  # Rotate right
            self.ship['angle'] = (self.ship['angle'] + 90) % 360
        elif action == 3:  # Thrust forward
            rad = math.radians(self.ship['angle'])
            self.ship['vx'] += math.cos(rad)
            self.ship['vy'] += math.sin(rad)
        elif action == 4:  # Shoot
            rad = math.radians(self.ship['angle'])
            self.bullets.append({'x': self.ship['x'], 'y': self.ship['y'],
                                 'vx': math.cos(rad) * 4 ,
                                 'vy': math.sin(rad) * 4 })
        
        # Update ship position
        new_x = (self.ship['x'] + self.ship['vx']) % self.width
        new_y = (self.ship['y'] + self.ship['vy']) % self.height
        
        # Check if the ship wraps around the screen
        wrapped = abs(new_x - old_x) > self.width / 2 or abs(new_y - old_y) > self.height / 2
        
        if not wrapped:
            for asteroid in self.asteroids:
                ax, ay = int(asteroid['x']), int(asteroid['y'])
                shape = asteroid['shape']
                for dy, row in enumerate(shape):
                    for dx, char in enumerate(row):
                        if char == 'O' and self.line_intersects_point(old_x, old_y, new_x, new_y, ax + dx, ay + dy):
                            self.lives -= 1
                            reward-=50
                            # return self.render(), -100, True
        
        self.ship['x'], self.ship['y'] = new_x, new_y
        
        # Update asteroids
        for asteroid in self.asteroids:
            asteroid['x'] = (asteroid['x'] + asteroid['vx']) % self.width
            asteroid['y'] = (asteroid['y'] + asteroid['vy']) % self.height
        
        # Update bullets
        new_bullets = []
        for bullet in self.bullets:
            bullet['x'] += bullet['vx']
            bullet['y'] += bullet['vy']
            if 0 <= bullet['x'] < self.width and 0 <= bullet['y'] < self.height:
                new_bullets.append(bullet)
        self.bullets = new_bullets
        
        # Check for collisions
        for bullet in self.bullets:
            for asteroid in self.asteroids:
                ax, ay = int(asteroid['x']), int(asteroid['y'])
                shape = asteroid['shape']
                for dy, row in enumerate(shape):
                    for dx, char in enumerate(row):
                        if char == 'O' and int(bullet['x']) == ax + dx and int(bullet['y']) == ay + dy:
                            self.asteroids.remove(asteroid)
                            self.bullets.remove(bullet)
                            reward += 10
                            break
        
        if not self.asteroids:
            self.done = True
            reward += 50
        
        return self.render(), reward, self.done, False, {}
    
    def render(self):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        ship_symbols = {0: '>', 90: 'v', 180: '<', 270: '^'}
        grid[int(self.ship['y'])][int(self.ship['x'])] = ship_symbols.get(self.ship['angle'], '>')
        
        for asteroid in self.asteroids:
            ax, ay = int(asteroid['x']), int(asteroid['y'])
            shape = asteroid['shape']
            for dy, row in enumerate(shape):
                for dx, char in enumerate(row):
                    if 0 <= ax + dx < self.width and 0 <= ay + dy < self.height:
                        grid[ay + dy][ax + dx] = char
        
        for bullet in self.bullets:
            grid[int(bullet['y'])][int(bullet['x'])] = '*'
        
        # Add border
        border_row = ['#' for _ in range(self.width + 2)]
        grid = [['#'] + row + ['#'] for row in grid]
        grid.insert(0, border_row)
        grid.append(border_row)
        
        return '\n'.join([''.join(row) for row in grid])
    def line_intersects_point(self, x1, y1, x2, y2, px, py):
        """
        Check if a line segment between (x1,y1) and (x2,y2) passes through
        or very close to the point (px,py).
        """
        # Convert to integers for consistent grid-based collision
        x1, y1, x2, y2, px, py = int(x1), int(y1), int(x2), int(y2), int(px), int(py)
        
        # If start and end are the same, just check direct collision
        if x1 == x2 and y1 == y2:
            return x1 == px and y1 == py
        
        # Check if the point is one of the endpoints
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):
            return True
            
        # Calculate distances
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # If line is too short, just check endpoints
        if line_length < 2:
            return False
            
        # Simple line rasterization to check all cells the line passes through
        if abs(x2 - x1) > abs(y2 - y1):
            # Line is more horizontal
            if x1 > x2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                
            for x in range(x1, x2 + 1):
                t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                y = y1 + t * (y2 - y1)
                if int(y) == py and x == px:
                    return True
        else:
            # Line is more vertical
            if y1 > y2:
                x1, x2 = x2, x1
                y1, y2 = y2, y1
                
            for y in range(y1, y2 + 1):
                t = (y - y1) / (y2 - y1) if y2 != y1 else 0
                x = x1 + t * (x2 - x1)
                if int(x) == px and y == py:
                    return True
                    
        return False
    
# Example usage
if __name__ == "__main__":
    game = AsciiAsteroids()
    print(game.reset(seed=42))
    done = False
    while not done:
        x = input("action: ")

        if (x=='0' or x=='1' or x=='2' or x=='3' or x=='4'):
            out = game.step(int(x))
            print(out[0])
            done = out[2]