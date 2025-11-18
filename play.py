import gymnasium as gym
from stable_baselines3 import DQN

# Importe a classe do ambiente
from dino_env import DinoEnv 

# --- Carregar e Executar o Modelo Treinado ---

# 1. Carregue o modelo salvo
# (Troque o nome do arquivo se quiser usar um checkpoint específico)
MODEL_PATH = "dino_dqn_final.zip" 

# 2. Crie o ambiente, mas desta vez com render_mode="human"
env = DinoEnv(render_mode="human")

# 3. Carregue o cérebro no agente
model = DQN.load(MODEL_PATH, env=env)

print("--- Iniciando a Demonstração ---")

# Loop principal do jogo
obs, info = env.reset()
score_total = 0
num_partidas = 0

while num_partidas < 10: # Jogue 10 partidas de demonstração
    # 'deterministic=True' faz a IA escolher a *melhor* ação, sem explorar
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    score_total += reward

    if terminated or truncated:
        print(f"Partida {num_partidas + 1} finalizada. Score (recompensa): {score_total:.2f}")
        score_total = 0
        num_partidas += 1
        obs, info = env.reset()

env.close()
print("--- Demonstração Concluída ---")