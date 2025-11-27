import gymnasium as gym
from stable_baselines3 import PPO
import os

# Importe a classe do ambiente
from dino_env import DinoEnv 

# --- ONDE PROCURAR O CÉREBRO? ---
# Lista de tentativas, do melhor para o pior
possible_paths = [
    "./dino_dqn_checkpoints/best_model/best_model.zip", # 1. O Campeão (Checkpoint)
    "./dino_dqn_checkpoints/dino_final_backup.zip",     # 2. O Final (Backup do treino)
    "dino_dqn_final.zip"                                  # 3. Antigo (Raiz)
]

model_path = ""

print("--- PROCURANDO CÉREBRO TREINADO ---")
for path in possible_paths:
    if os.path.exists(path):
        print(f"✅ Encontrado: {path}")
        model_path = path
        break
    else:
        print(f"❌ Não encontrado: {path}")

if not model_path:
    print("\nERRO CRÍTICO: Nenhum arquivo de modelo encontrado!")
    print("Verifique se a pasta 'dino_dqn_checkpoints' foi criada.")
    exit()

# --- EXECUÇÃO ---

# Cria o ambiente visual
env = DinoEnv(render_mode="human")

# Carrega o cérebro PPO
try:
    model = PPO.load(model_path, env=env)
except Exception as e:
    print(f"Erro ao carregar: {e}")
    exit()

print("\n--- INICIANDO O JOGO ---")
print("A IA vai jogar agora. Pressione CTRL+C no terminal para parar.")

obs, info = env.reset()
score_total = 0
num_partidas = 0

while True: # Joga infinitamente
    # deterministic=True faz a IA jogar sério (sem aleatoriedade)
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    score_total += reward

    if terminated or truncated:
        print(f"Partida {num_partidas + 1} acabou. Score: {score_total:.2f}")
        score_total = 0
        num_partidas += 1
        obs, info = env.reset()
        # Pequena pausa para respirar entre mortes (opcional)
        # pygame.time.delay(500) 

env.close()