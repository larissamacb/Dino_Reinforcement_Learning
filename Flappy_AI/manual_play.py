import pygame
# Importa as constantes de tamanho também para abrir a janela
from game import FlappyGame, SCREEN_WIDTH, SCREEN_HEIGHT 

# 1. Liga o Pygame ANTES de tudo
pygame.init()

# 2. Cria a janela ANTES de carregar o jogo (necessário para carregar imagens)
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird - Modo Manual")

# 3. AGORA sim cria o jogo (que vai carregar as imagens)
game = FlappyGame()
game.screen = screen # Avisa o jogo para usar essa tela

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

    # O jogo roda a lógica
    # (O jogo retorna recompensa, mas ignoramos isso no modo manual)
    game.step(action)
    
    # O jogo desenha na tela que criamos
    game.render()
    
    if game.game_over:
        print(f"Você morreu! Score final: {game.score}")
        game.reset()

    clock.tick(30)

pygame.quit()