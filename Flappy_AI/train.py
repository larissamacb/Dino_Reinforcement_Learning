from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
from flappy_env import FlappyEnv

LOG_DIR = "./flappy_logs/"
MODEL_DIR = "./flappy_models/"
BEST_MODEL_DIR = "./flappy_models/best_model/"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

env = DummyVecEnv([lambda: FlappyEnv(render_mode=None)])

eval_callback = EvalCallback(
    env, 
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR, 
    eval_freq=10000, 
    deterministic=True, 
    render=False
)

eval_callback.best_mean_reward = 175.0 

# Modelo PPO
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003)

print("--- TREINANDO FLAPPY BIRD ---")

model.learn(total_timesteps=5000000, callback=eval_callback) 

print("Treino finalizado.")