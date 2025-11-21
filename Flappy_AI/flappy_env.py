import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import FlappyGame, SCREEN_WIDTH, SCREEN_HEIGHT, PIPE_GAP

class FlappyEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.game = FlappyGame()
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(2)
        
        # Observação Normalizada
        self.observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def _get_obs(self):
        next_pipe = None
        for pipe in self.game.pipes:
            if pipe['top'].right > self.game.bird_rect.left:
                next_pipe = pipe
                break
        
        if next_pipe is None:
            return np.array([0.5, 0.0, 1.0, 0.5], dtype=np.float32)

        # Normalização
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
            self.game.render()

        return obs, reward, terminated, False, self._get_info()

    def close(self):
        pass