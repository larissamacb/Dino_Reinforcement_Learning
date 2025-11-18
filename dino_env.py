import gymnasium as gym
from gymnasium import spaces
import pygame
import random
import os
import numpy

# Importa as classes e variáveis do arquivo de jogo que VOCÊ baixou
from jogo.chromedino import (
    Dinosaur, 
    SmallCactus, 
    LargeCactus, 
    Bird, 
    Cloud, 
    BG, 
    SCREEN_WIDTH, 
    SCREEN_HEIGHT,
    FONT_COLOR,
    SMALL_CACTUS,
    LARGE_CACTUS,
    BIRD
)

# --- Classe do Ambiente Gymnasium ---

class DinoEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.game_over = False
        
        # --- 🚀 MUDANÇA 1: "OLHOS" MELHORES ---
        # Ações: 0 = Correr, 1 = Pular, 2 = Abaixar
        self.action_space = spaces.Discrete(3)

        # Observações (Estado): 5 VALORES AGORA
        # [pos_y_dino, dist_obst, altura_obst, tipo_obst, vel_jogo]
        # Tipo: 0.0 = Cacto, 1.0 = Pássaro
        low = numpy.array([
            0,                 # pos_y_dino
            0,                 # dist_obst
            0,                 # altura_obst
            0,                 # tipo_obst
            0                  # vel_jogo
        ], dtype=numpy.float32)
        
        high = numpy.array([
            SCREEN_HEIGHT,     # pos_y_dino
            SCREEN_WIDTH * 2,  # dist_obst
            SCREEN_HEIGHT,     # altura_obst
            1,                 # tipo_obst
            100                # vel_jogo
        ], dtype=numpy.float32)
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=numpy.float32)

    def _get_obs(self):
        """Retorna a observação atual (estado com 5 valores)."""
        
        # Valor 1: Posição Y do Dinossauro
        dino_y = float(self.player.dino_rect.y)

        # Se não houver obstáculos, retorne um estado "seguro"
        if not self.obstacles:
            # dino_y, dist, altura, tipo(cacto), vel
            return numpy.array([dino_y, SCREEN_WIDTH, 0, 0.0, self.game_speed], dtype=numpy.float32)

        # Encontra o próximo obstáculo
        player_x = self.player.dino_rect.x + self.player.dino_rect.width
        valid_obstacles = [o for o in self.obstacles if o.rect.x + o.rect.width > player_x]
        
        if not valid_obstacles:
            return numpy.array([dino_y, SCREEN_WIDTH, 0, 0.0, self.game_speed], dtype=numpy.float32)

        next_obstacle = min(valid_obstacles, key=lambda o: o.rect.x)
        
        # Valor 2: Distância
        distance = float(next_obstacle.rect.x - player_x)
        # Valor 3: Altura do obstáculo
        obstacle_height = float(next_obstacle.rect.y)
        # Valor 4: Tipo do obstáculo
        if isinstance(next_obstacle, Bird):
            obstacle_type = 1.0 # É um Pássaro
        else:
            obstacle_type = 0.0 # É um Cacto
        # Valor 5: Velocidade
        speed = self.game_speed

        return numpy.array([dino_y, distance, obstacle_height, obstacle_type, speed], dtype=numpy.float32)

    def _get_info(self):
        return {"score": self.points, "speed": self.game_speed, "passed": len(self.passed_obstacles)}

    def reset(self, seed=None, options=None):
        """Reinicia o jogo."""
        super().reset(seed=seed)
        
        if self.render_mode == "human" and self.screen is None:
            pygame.display.set_caption("Dino IA - Treinamento (PRO)")
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font("freesansbold.ttf", 20)

        # --- 🚀 MUDANÇA 2: "MOTIVAÇÃO" MELHOR ---
        # Reseta a lista de obstáculos passados
        self.passed_obstacles = set()

        self.player = Dinosaur()
        self.cloud = Cloud()
        self.obstacles = []
        self.game_speed = 20
        self.points = 0
        self.x_pos_bg = 0
        self.y_pos_bg = 380
        self.game_over = False

        observation = self._get_obs()
        info = self._get_info()
        return observation, info
    
    def _background(self):
        """Função privada para desenhar o chão rolando"""
        if self.screen is None: # Adicionado para segurança no modo cego
            return
            
        image_width = BG.get_width()
        self.screen.blit(BG, (self.x_pos_bg, self.y_pos_bg))
        self.screen.blit(BG, (image_width + self.x_pos_bg, self.y_pos_bg))
        if self.x_pos_bg <= -image_width:
            self.x_pos_bg = 0
        self.x_pos_bg -= self.game_speed

    def _score(self):
        """Função privada para atualizar e desenhar o placar"""
        self.points += 1
        if self.points % 100 == 0:
            if self.game_speed < 40: # Limite de velocidade
                self.game_speed += 1

        # Esta parte só roda se estivermos no modo "human" e o placar existir
        if self.render_mode == "human" and self.font is not None:
            text = self.font.render("Points: " + str(self.points), True, FONT_COLOR)
            textRect = text.get_rect()
            textRect.center = (900, 40)
            self.screen.blit(text, textRect)

    def step(self, action):
        """Executa um passo no tempo (1 frame)"""
        if self.game_over:
            return self._get_obs(), 0, True, False, self._get_info()
        
        # --- 1. EXECUTAR A AÇÃO (COM 3 OPÇÕES) ---
        userInput = {pygame.K_UP: False, pygame.K_DOWN: False, pygame.K_SPACE: False}
        if action == 1:
            userInput[pygame.K_UP] = True  # Pular
        elif action == 2:
            userInput[pygame.K_DOWN] = True # Abaixar

        self.player.update(userInput)
        self.cloud.update(self.game_speed)

        # Gera obstáculos (COM PÁSSAROS)
        if len(self.obstacles) == 0:
            if random.randint(0, 2) == 0:
                self.obstacles.append(SmallCactus(SMALL_CACTUS))
            elif random.randint(0, 2) == 1:
                self.obstacles.append(LargeCactus(LARGE_CACTUS))
            elif random.randint(0, 2) == 2:
                self.obstacles.append(Bird(BIRD))

        # Atualiza os obstáculos
        obstacles_to_remove = []
        for obstacle in self.obstacles:
            obstacle.update(self.game_speed)
            if obstacle.rect.x < -obstacle.rect.width:
                obstacles_to_remove.append(obstacle)
        
        for obstacle in obstacles_to_remove:
            self.obstacles.pop(self.obstacles.index(obstacle))

        self._score()

        # --- 3. Verificar Colisões ---
        terminated = False
        for obstacle in self.obstacles:
            if self.player.dino_rect.colliderect(obstacle.rect):
                self.game_over = True
                terminated = True
                break

        # --- 4. CALCULAR RECOMPENSA (COM BÔNUS) ---
        reward = 0
        if self.game_over:
            reward = -30 # Punição GRANDE por morrer
        else:
            reward = 0.1 # Recompensa pequena por sobreviver
        
        # --- 🚀 MUDANÇA 2 (CONTINUAÇÃO): BÔNUS POR SUCESSO ---
        player_x = self.player.dino_rect.x
        for obs in self.obstacles:
            # Se o obstáculo não foi passado E agora está atrás do dino
            if obs not in self.passed_obstacles and obs.rect.x < player_x:
                self.passed_obstacles.add(obs)
                reward += 50.0  # BÔNUS!
                
        # --- 5. Pegar novas informações ---
        observation = self._get_obs()
        info = self._get_info()
        truncated = False 

        # --- 6. Renderizar ---
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _render_frame(self):
        if self.screen is None: return
        pygame.event.pump()
        self.screen.fill((255, 255, 255))
        self.player.draw(self.screen)
        for obstacle in self.obstacles: obstacle.draw(self.screen)
        self._background()
        self.cloud.draw(self.screen)
        self._score()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None