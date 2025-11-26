import pygame
from game import FlappyGame, SCREEN_WIDTH, SCREEN_HEIGHT 

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Modo Manual")

game = FlappyGame()
game.screen = screen 

running = True
print("--- MODO MANUAL ---")
print("Pressione [ESPAÇO] para pular")

clock = pygame.time.Clock()

while running:
    action = 0 
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1 # Pular

    game.step(action)

    game.render()
    
    if game.game_over:
        print(f"Você morreu! Score final: {game.score}")
        game.reset()

    clock.tick(30)

pygame.quit()