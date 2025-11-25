import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from game import FlappyGame, SCREEN_WIDTH, SCREEN_HEIGHT, PIPE_GAP

class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = FlappyGame()
        self.render_mode = render_mode
        
        # Ações: 0 = Nada, 1 = Pular
        self.action_space = spaces.Discrete(2)

        # Observação Normalizada
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def _get_obs(self):
        # Pega o próximo cano à frente
        next_pipe = None
        for pipe in self.game.pipes:
            if pipe['top'].right > self.game.bird_rect.left:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            return np.array([0.5, 0.0, 1.0, 0.5], dtype=np.float32)

        # Normalização dos dados para a IA
        dist_x = (next_pipe['top'].left - self.game.bird_rect.right) / SCREEN_WIDTH
        gap_center_y = next_pipe['bottom'].top - (PIPE_GAP / 2)
        dist_y = (gap_center_y - self.game.bird_rect.centery) / SCREEN_HEIGHT
        bird_y = self.game.bird_rect.centery / SCREEN_HEIGHT
        vel = self.game.bird_vel / 20.0

        return np.array([dist_x, dist_y, bird_y, vel], dtype=np.float32)

    def _get_info(self):
        return {"score": self.game.score}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        return self._get_obs(), self._get_info()

    def step(self, action):
        reward = self.game.step(action)
        obs = self._get_obs()
        terminated = self.game.game_over
        
        if self.render_mode == "human":
            self._render_frame() # Chama nossa renderização customizada

        return obs, reward, terminated, False, self._get_info()

    def _render_frame(self):
        # Desenha o jogo normal primeiro
        self.game.render()
        
        # --- AQUI COMEÇA A VISUALIZAÇÃO DA IA ---
        screen = self.game.screen # Pega a tela do jogo
        if screen is None: return

        # 1. Achar o alvo (o próximo cano)
        target_pipe = None
        for pipe in self.game.pipes:
            if pipe['top'].right > self.game.bird_rect.left:
                target_pipe = pipe
                break
        
        if target_pipe:
            # Calcula o centro do buraco (Onde a IA quer chegar)
            gap_center_y = target_pipe['bottom'].top - (PIPE_GAP / 2)
            pipe_center_x = target_pipe['top'].centerx
            
            bird_center = self.game.bird_rect.center

            # 🔴 LINHA VERMELHA: O "Olhar" da IA
            # Conecta o pássaro ao objetivo. Representa Distância X e Distância Y.
            pygame.draw.line(screen, (255, 0, 0), bird_center, (pipe_center_x, gap_center_y), 3)
            
            # Desenha uma bolinha no alvo para ficar claro
            pygame.draw.circle(screen, (255, 0, 0), (int(pipe_center_x), int(gap_center_y)), 5)

            # 🔵 LINHA AZUL: A Velocidade Vertical
            # Se aponta pra baixo, o pássaro cai. Pra cima, ele sobe.
            # Multiplicamos por 10 para a linha ficar visível
            vel_vector_end = (bird_center[0], bird_center[1] + (self.game.bird_vel * 10))
            pygame.draw.line(screen, (0, 0, 255), bird_center, vel_vector_end, 4)

        # Atualiza a tela com os desenhos novos
        pygame.display.update()

    def close(self):
        pass