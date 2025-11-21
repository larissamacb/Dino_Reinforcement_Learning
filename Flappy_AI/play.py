import pygame
from stable_baselines3 import PPO
from flappy_env import FlappyEnv

# Liga o Pygame
pygame.init()

# Carrega o MELHOR modelo salvo pelo EvalCallback
model_path = "./flappy_models/best_model/best_model.zip"
env = FlappyEnv(render_mode="human")

print(f"Carregando o CAMPEÃO: {model_path}...")
try:
    model = PPO.load(model_path, env=env)
except FileNotFoundError:
    print("ERRO: Arquivo 'best_model.zip' não encontrado.")
    print("Você rodou o train.py por tempo suficiente?")
    exit()

obs, _ = env.reset()
current_score = 0

print("\n--- JOGANDO ---")
print("Pressione [ESC] para sair.")

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            env.close()
            exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                env.close()
                exit()

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)

    # Atualiza Score
    if "score" in info:
        current_score = info["score"]

    if done:
        print(f"O pássaro morreu. Score Final: {current_score}")
        obs, _ = env.reset()
        current_score = 0
    
    env.game.clock.tick(30)