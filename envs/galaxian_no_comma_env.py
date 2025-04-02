import numpy as np
import random

class AsciiGalaxian:
    metadata = {"render_modes": ["human"], "render_fps": 20}
    def __init__(self, size=40, lives=100, ammo_max=5, ammo_replenish_rate=4, enemy_fire_chance = 0.01):
        self.size = size
        self.width = self.size
        self.height = int(self.size/2)
        self.lives = lives
        self.level = 1
        self.ammo_max = ammo_max
        self.ammo_replenish_rate = ammo_replenish_rate
        self.enemy_fire_chance = enemy_fire_chance
        self.reset(seed=0)
    
    def reset(self, seed=0):
        #random.seed(seed)
        self.ship = {'x': self.width // 2, 'y': self.height - 2}
        self.spawn_enemies()
        self.bullets = []
        self.enemy_bullets = []
        self.done = False
        self.ammo = self.ammo_max
        self.frame_count = 0
        return self.render(), {}
    
    def spawn_enemies(self):
        self.enemies = [{'x': x, 'y': y, 'direction': 1, 'charging': False} 
                        for x in range(3, self.width - 3, 3) for y in range(2, 6)]
        self.charging_enemies = []
    
    def step(self, action):
        if self.done:
            return self.render(), 0, True
        
        reward = 0
        self.frame_count += 1
        
        if self.frame_count % self.ammo_replenish_rate == 0 and self.ammo < self.ammo_max:
            self.ammo += 1
        
        if action == 1 and self.ship['x'] > 1:  # Move left
            self.ship['x'] -= 1
        elif action == 2 and self.ship['x'] < self.width - 2:  # Move right
            self.ship['x'] += 1
        elif action == 3 and self.ammo > 0:  # Shoot
            self.bullets.append({'x': self.ship['x'], 'y': self.ship['y']})
            self.ammo -= 1

        # Update bullets
        self.bullets = [{'x': b['x'], 'y': b['y'] - 1} for b in self.bullets if b['y'] > 0]
        
        # Check collisions
        for bullet in self.bullets:
            for enemy in self.enemies[:]:
                if bullet['x'] == enemy['x'] and bullet['y'] == enemy['y']:
                    self.enemies.remove(enemy)
                    if enemy in self.charging_enemies:
                        self.charging_enemies.remove(enemy)
                    self.bullets.remove(bullet)
                    reward += 10
                    break
        # Select up to two charging enemies
        if len(self.charging_enemies) < 3:
            available_enemies = [e for e in self.enemies if not e['charging']]
            if available_enemies:
                enemy = random.choice(available_enemies)
                enemy['charging'] = True
                if enemy['x']>self.ship['x']:
                    enemy['target_x'] = self.ship['x']-2
                elif enemy['x']<self.ship['x']:
                    enemy['target_x'] = self.ship['x']+2
                else:
                    enemy['target_x'] = self.ship['x']
                
                self.charging_enemies.append(enemy)
        
        # Update enemies
        for enemy in self.enemies[:]:
            if enemy['charging']:
                if enemy['x'] < enemy['target_x']:
                    enemy['x'] += 1
                elif enemy['x'] > enemy['target_x']:
                    enemy['x'] -= 1

                enemy['y'] += 1
                
                # Check if charging enemy hits the player
                if enemy['x'] == self.ship['x'] and enemy['y'] == self.ship['y']:
                    self.lives -= 1
                    self.enemies.remove(enemy)
                    self.charging_enemies.remove(enemy)
                    if self.lives == 0:
                        self.done = True
                        return self.render(), -100, True, False, {}
                    else:
                        return self.render(), -50, False, False, {}
                
                if enemy['y'] > self.height - 1:
                    enemy['y'] = 2  # Respawn at the top
                    enemy['x'] = random.randint(3, self.width - 3)  # Randomize respawn position
                    enemy['charging'] = False
                    self.charging_enemies.remove(enemy)
            else:
                enemy['x'] += enemy['direction']
                if enemy['x'] <= 1 or enemy['x'] >= self.width - 2:
                    enemy['direction'] *= -1
                    enemy['y'] += 1
            if enemy["charging"]:
                if random.random() < self.enemy_fire_chance*20:
                    self.enemy_bullets.append({'x': enemy['x'], 'y': enemy['y']})
            else:
                if random.random() < self.enemy_fire_chance:
                    self.enemy_bullets.append({'x': enemy['x'], 'y': enemy['y']})
        
        # Update enemy bullets
        self.enemy_bullets = [{'x': b['x'], 'y': b['y'] + 1} for b in self.enemy_bullets if 0 <= b['y'] + 1 < self.height]
        
        # Check collisions
        for bullet in self.bullets:
            for enemy in self.enemies[:]:
                if bullet['x'] == enemy['x'] and bullet['y'] == enemy['y']:
                    self.enemies.remove(enemy)
                    if enemy in self.charging_enemies:
                        self.charging_enemies.remove(enemy)
                    self.bullets.remove(bullet)
                    reward += 10
                    break
        
        for bullet in self.enemy_bullets:
            if bullet['x'] == self.ship['x'] and bullet['y'] == self.ship['y']:
                self.lives -= 1
                if self.lives == 0:
                    self.done = True
                    return self.render(), -100, True, False, {}
                else:
                    return self.render(), -50, False, False, {}
        
        if not self.enemies:
            self.level += 1
            self.spawn_enemies()
            reward += 50
        
        return self.render(), reward, self.done, False, {}
    
    def render(self):
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        grid[self.ship['y']][self.ship['x']] = 'A'
        
        for enemy in self.enemies:
            grid[enemy['y']][enemy['x']] = 'M'
        
        for bullet in self.bullets:
            if 0 <= bullet['y'] < self.height and 0 <= bullet['x'] < self.width:
                grid[bullet['y']][bullet['x']] = '*'
        
        for bullet in self.enemy_bullets:
            if 0 <= bullet['y'] < self.height and 0 <= bullet['x'] < self.width:
                grid[bullet['y']][bullet['x']] = '!'
        
        border_row = ['#' for _ in range(self.width + 2)]
        grid = [['#'] + row + ['#'] for row in grid]
        grid.insert(0, border_row)
        grid.append(border_row)
        
        return '\n'.join([''.join(row) for row in grid])
    
# Example usage
# Example usage
if __name__ == "__main__":
    game = AsciiGalaxian()
    print(game.reset(seed=42))
    done = False
    while not done:
        x = input("action: ")

        if (x=='0' or x=='1' or x=='2' or x=='3'):
            out = game.step(int(x))
            print(out[0])
            done = out[2]

