import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
from dino_env import DinoEnv 

LOG_DIR = "./dino_tensorboard/"
MODEL_DIR = "./dino_dqn_checkpoints/"
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")

# Vamos colocar um número alto, mas ele vai parar antes se ficar bom
TOTAL_TIMESTEPS = 1_000_000 

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

print("--- TREINAMENTO DINO (META: 5000 PONTOS) ---")

env = DummyVecEnv([lambda: DinoEnv(render_mode=None)])
policy_kwargs = dict(net_arch=[256, 256])

# --- LÓGICA DE CHECKPOINT ---
if os.path.exists(BEST_MODEL_PATH):
    print(f"✅ Encontrei um campeão em: {BEST_MODEL_PATH}")
    print("🔄 Continuando para ver se ele bate 5k...")
    model = PPO.load(BEST_MODEL_PATH, env=env, tensorboard_log=LOG_DIR)
else:
    print("✨ Começando do ZERO.")
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, 
                tensorboard_log=LOG_DIR, learning_rate=0.0003, ent_coef=0.01, 
                batch_size=64, n_steps=2048, gamma=0.99)

# --- A REGRA DE PARADA ---
# Se a recompensa média nos testes passar de 4000 (margem de segurança para 5k)
callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=4000, verbose=1)

eval_callback = EvalCallback(
    env, 
    callback_on_new_best=callback_on_best, # <-- Chama a parada se bater o recorde
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR, 
    eval_freq=20000, 
    deterministic=True, 
    render=False,
    n_eval_episodes=3
)

print("🚀 Iniciando...")
print("Se a IA ficar 'Imortal' (5k+ pontos), o treino para sozinho.")

try:
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback, reset_num_timesteps=False, progress_bar=True)
    print("✅ Treino finalizado! A IA atingiu o limite ou acabou o tempo.")
except KeyboardInterrupt:
    print("\n⚠️ Pausado.")

model.save(os.path.join(MODEL_DIR, "dino_final_backup"))