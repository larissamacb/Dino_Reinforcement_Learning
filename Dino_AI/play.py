import gymnasium as gym
from stable_baselines3 import PPO 

from dino_env import DinoEnv 

MODEL_PATH = "dino_dqn_final.zip" 

env = DinoEnv(render_mode="human")

model = PPO.load(MODEL_PATH, env=env)

print("--- Iniciando a Demonstração PPO ---")

obs, info = env.reset()
score_total = 0
num_partidas = 0

while num_partidas < 10:
    action, _states = model.predict(obs, deterministic=True)
    
    obs, reward, terminated, truncated, info = env.step(action)
    score_total += reward

    if terminated or truncated:
        print(f"Partida {num_partidas + 1}: Score {score_total:.2f}")
        score_total = 0
        num_partidas += 1
        obs, info = env.reset()

env.close()