from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
from flappy_env import FlappyEnv

# Configuração de Pastas
LOG_DIR = "./flappy_logs/"
MODEL_DIR = "./flappy_models/"
BEST_MODEL_DIR = "./flappy_models/best_model/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Ambiente de Treino
env = DummyVecEnv([lambda: FlappyEnv(render_mode=None)])

# --- CALLBACK PARA SALVAR O MELHOR ---
# A cada 10.000 passos, ele joga 5 vezes para testar.
eval_callback = EvalCallback(
    env, 
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR, 
    eval_freq=10000, 
    deterministic=True, 
    render=False
)

# --- A MUDANÇA ESTÁ AQUI ---
# Define que a recompensa média mínima para salvar é 175.0
# Isso equivale a aproximadamente 50 canos.
# Se a IA não conseguir isso, a pasta best_model continuará vazia ou manterá o antigo.
eval_callback.best_mean_reward = 175.0 

# Modelo PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003)

print("--- TREINANDO FLAPPY BIRD (Busca pela Perfeição) ---")
print("O script só vai salvar o modelo se ele fizer média > 50 canos (Reward 175).")
print("Treino aumentado para 5 milhões de passos para dar tempo.")

# Aumentado para 5 milhões de passos
model.learn(total_timesteps=5000000, callback=eval_callback) 

print("Treino finalizado.")