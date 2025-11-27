import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import os
import numpy

from jogo.chromedino import (
    Dinosaur, SmallCactus, LargeCactus, Bird, Cloud, BG, 
    SCREEN_WIDTH, SCREEN_HEIGHT, FONT_COLOR, 
    SMALL_CACTUS, LARGE_CACTUS, BIRD
)

class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        
        # 3 Ações
        self.action_space = spaces.Discrete(3)

        # Observação FÍSICA
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=numpy.float32)
        self.last_action = 0

    def _get_obs(self):
        dino_y = self.player.dino_rect.y / SCREEN_HEIGHT

        if not self.obstacles:
            return numpy.array([dino_y, 1.0, 0.0, 0.0], dtype=numpy.float32)

        player_x = self.player.dino_rect.x + self.player.dino_rect.width
        valid_obstacles = [o for o in self.obstacles if o.rect.x + o.rect.width > player_x]
        
        if not valid_obstacles:
            return numpy.array([dino_y, 1.0, 0.0, 0.0], dtype=numpy.float32)

        target = min(valid_obstacles, key=lambda o: o.rect.x)
        
        dist_pixels = target.rect.x - player_x
        if dist_pixels < 0: dist_pixels = 0
        
        # Normaliza o tempo (dividindo por 50.0 para sensibilidade)
        time_to_impact = (dist_pixels / self.game_speed) / 50.0 
        time_to_impact = max(0.0, min(1.0, time_to_impact))

        obstacle_height = target.rect.y / SCREEN_HEIGHT
        obstacle_type = 1.0 if isinstance(target, Bird) else 0.0

        return numpy.array([dino_y, time_to_impact, obstacle_height, obstacle_type], dtype=numpy.float32)

    def _get_info(self):
        return {"score": self.points}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.render_mode == "human" and self.screen is None:
            pygame.display.set_caption("Dino IA - Modo Infinito")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("freesansbold.ttf", 20)

        self.player = Dinosaur()
        self.cloud = Cloud()
        self.obstacles = []
        self.game_speed = 20
        self.points = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.game_over = False
        self.passed_obstacles = set()
        return self._get_obs(), self._get_info()

    def step(self, action):
        if self.game_over: return self._get_obs(), 0, True, False, self._get_info()
        self.last_action = action
        
        userInput = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_SPACE: False}
        if action == 1: userInput[pygame.K_UP] = True
        elif action == 2: userInput[pygame.K_DOWN] = True

        self.player.update(userInput)
        self.cloud.update(self.game_speed)

        if len(self.obstacles) == 0:
            r = random.randint(0, 2)
            if r == 0: self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif r == 1: self.obstacles.append(LargeCactus(LARGE_CACTUS))
            elif r == 2: self.obstacles.append(Bird(BIRD))

        obstacles_to_remove = []
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
            if obstacle.rect.x < -obstacle.rect.width: obstacles_to_remove.append(obstacle)
        for o in obstacles_to_remove: self.obstacles.remove(o)

        self.points += 1
        if self.points % 100 == 0 and self.game_speed < 40: self.game_speed += 1
        
        terminated = False
        for obstacle in self.obstacles:
            if self.player.dino_rect.colliderect(obstacle.rect):
                self.game_over = True
                terminated = True
                break

        # --- 🚀 META DE 5000 PONTOS ---
        truncated = False
        # Só encerra o jogo se bater a meta E NÃO ESTIVERMOS ASSISTINDO
        if self.points >= 5000 and self.render_mode != "human":
            self.game_over = True
            terminated = True
            # Bônus GIGANTE por "zerar" o treino
            return self._get_obs(), 200.0, terminated, truncated, self._get_info()

        # --- SISTEMA DE RECOMPENSA ---
        reward = 0.1 
        obs = self._get_obs()
        time_to_impact = obs[1]

        # PUNIÇÃO POR PULAR CEDO (Anti-Ansiedade)
        if action == 1 and time_to_impact > 0.4:
            reward -= 0.5

        # INCENTIVO DE PRECISÃO
        if action == 1 and time_to_impact <= 0.4:
            reward += 0.1

        if self.game_over: reward = -10.0

        # Bônus por passar obstáculo
        player_x = self.player.dino_rect.x
        for obs_obj in self.obstacles:
            if obs_obj not in self.passed_obstacles and obs_obj.rect.x < player_x:
                self.passed_obstacles.add(obs_obj)
                reward += 5.0

        if self.render_mode == "human": self._render_frame()
        
        return obs, reward, terminated, truncated, self._get_info()

    def _render_frame(self):
        if self.screen is None: return
        self.screen.fill((255, 255, 255))
        image_width = BG.get_width()
        self.screen.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width: self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed
        
        self.player.draw(self.screen)
        self.cloud.draw(self.screen)
        for obs in self.obstacles: obs.draw(self.screen)
        
        # Debug
        player_x = self.player.dino_rect.x + self.player.dino_rect.width
        player_y = self.player.dino_rect.y
        valid_obstacles = [o for o in self.obstacles if o.rect.x + o.rect.width > player_x]
        if valid_obstacles:
            target = min(valid_obstacles, key=lambda o: o.rect.x)
            pygame.draw.line(self.screen, (255, 0, 0), (player_x, player_y), (target.rect.x, target.rect.y), 2)
            
            if self.font:
                acao = ["CORRER", "PULAR", "AGACHAR"][self.last_action]
                obs = self._get_obs()
                tempo = obs[1]
                txt = self.font.render(f"IA: {acao} | Tempo: {tempo:.2f}", True, (255,0,0))
                self.screen.blit(txt, (50, 50))

        if self.font:
            text = self.font.render(f"Points: {self.points}", True, FONT_COLOR)
            self.screen.blit(text, (900, 40))
        
        pygame.display.update()
        pygame.event.pump()
        self.clock.tick(30) 

    def close(self):
        if self.screen:
            pygame.display.quit()
            pygame.quit()
            self.screen = None