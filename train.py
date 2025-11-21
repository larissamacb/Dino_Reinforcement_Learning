import gymnasium as gym
from stable_baselines3 import PPO 
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from dino_env import DinoEnv 

LOG_DIR = "./dino_tensorboard/"
MODEL_SAVE_PATH = "./dino_dqn_checkpoints/"
TOTAL_TIMESTEPS = 1_000_000 
CHECKPOINT_FREQ = 100000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("--- TREINAMENTO PPO + ENTROPIA ---")
env = DummyVecEnv([lambda: DinoEnv(render_mode=None)])

policy_kwargs = dict(net_arch=[128, 128])

model = PPO(
    "MlpPolicy", 
    env, 
    policy_kwargs=policy_kwargs,
    verbose=1, 
    tensorboard_log=LOG_DIR,
    learning_rate=0.0003,
    ent_coef=0.01, # <-- ISSO FORÇA A IA A NÃO VICIAR EM UMA AÇÃO
    batch_size=64,
    n_steps=2048,
    gamma=0.99,
)

checkpoint_callback = CheckpointCallback(save_freq=CHECKPOINT_FREQ, save_path=MODEL_SAVE_PATH, name_prefix="dino_ppo")

try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
except KeyboardInterrupt:
    print("\nParado pelo usuário.")

model.save("dino_dqn_final")
print("Treino salvo.")