import pygame
import random
import os

# Configurações
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
PIPE_GAP = 100
BASE_Y = SCREEN_HEIGHT * 0.79
GRAVITY = 0.25
FLAP_STRENGTH = -4.5
PIPE_SPEED = 4

class FlappyGame:
    def __init__(self):
        self.screen = None
        self.clock = pygame.time.Clock()
        
        # Carrega Assets
        self.bg_img = pygame.image.load(os.path.join("assets", "sprites", "background-day.png"))
        self.base_img = pygame.image.load(os.path.join("assets", "sprites", "base.png"))
        self.bird_imgs = [
            pygame.image.load(os.path.join("assets", "sprites", "bluebird-downflap.png")),
            pygame.image.load(os.path.join("assets", "sprites", "bluebird-midflap.png")),
            pygame.image.load(os.path.join("assets", "sprites", "bluebird-upflap.png"))
        ]
        self.pipe_img = pygame.image.load(os.path.join("assets", "sprites", "pipe-green.png"))
        
        self.reset()

    def reset(self):
        self.bird_y = int(SCREEN_HEIGHT / 2)
        self.bird_vel = 0
        self.bird_rect = self.bird_imgs[0].get_rect(center=(50, self.bird_y))
        
        self.base_x = 0
        self.score = 0
        self.game_over = False
        
        self.pipes = []
        self.spawn_pipe()

    def spawn_pipe(self):
        height = random.randint(50, 250)
        bottom_pipe = self.pipe_img.get_rect(midtop=(SCREEN_WIDTH + 50, height + PIPE_GAP))
        top_pipe = self.pipe_img.get_rect(midbottom=(SCREEN_WIDTH + 50, height))
        self.pipes.append({'top': top_pipe, 'bottom': bottom_pipe, 'passed': False})

    def step(self, action):
        if action == 1:
            self.bird_vel = FLAP_STRENGTH
        
        self.bird_vel += GRAVITY
        self.bird_y += self.bird_vel
        self.bird_rect.centery = self.bird_y
        
        # Rotação visual
        self.bird_rotation = -self.bird_vel * 3
        
        # Move chão
        self.base_x -= PIPE_SPEED
        if self.base_x <= -48:
            self.base_x = 0

        # Move Canos
        for pipe in self.pipes:
            pipe['top'].centerx -= PIPE_SPEED
            pipe['bottom'].centerx -= PIPE_SPEED
        
        if self.pipes and self.pipes[-1]['top'].centerx < SCREEN_WIDTH - 150:
            self.spawn_pipe()
        
        if self.pipes and self.pipes[0]['top'].right < 0:
            self.pipes.pop(0)

        self.check_collision()
        
        reward = 0.1
        for pipe in self.pipes:
            if not pipe['passed'] and pipe['top'].centerx < self.bird_rect.centerx:
                self.score += 1
                pipe['passed'] = True
                reward = 1.0
        
        if self.game_over:
            reward = -1.0
            
        return reward

    def check_collision(self):
        if self.bird_rect.top <= -50 or self.bird_rect.bottom >= BASE_Y:
            self.game_over = True
            return

        for pipe in self.pipes:
            if self.bird_rect.colliderect(pipe['top']) or self.bird_rect.colliderect(pipe['bottom']):
                self.game_over = True
                return

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy AI")

        self.screen.blit(self.bg_img, (0, 0))

        for pipe in self.pipes:
            flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(flipped_pipe, pipe['top'])
            self.screen.blit(self.pipe_img, pipe['bottom'])

        self.screen.blit(self.base_img, (self.base_x, BASE_Y))

        bird_surface = self.bird_imgs[1]
        rotated_bird = pygame.transform.rotozoom(bird_surface, self.bird_rotation, 1)
        self.screen.blit(rotated_bird, self.bird_rect)

        pygame.display.update()