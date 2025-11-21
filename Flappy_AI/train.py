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
# A cada 10.000 passos, ele joga 5 vezes. 
# Se for a melhor média até agora, ele salva em 'best_model/best_model.zip'
eval_callback = EvalCallback(
    env, 
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR, 
    eval_freq=10000, 
    deterministic=True, 
    render=False
)

# Modelo PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003)

print("--- TREINANDO FLAPPY BIRD (Modo Campeão) ---")
print("O script vai salvar o MELHOR modelo automaticamente.")
print("Pode deixar rodar os 3 milhões. Se piorar no fim, o melhor estará salvo.")

# Treino Longo (Seguro porque temos backup do melhor momento)
model.learn(total_timesteps=3000000, callback=eval_callback) 

print("Treino finalizado.")