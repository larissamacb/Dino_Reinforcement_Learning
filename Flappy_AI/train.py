from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
import os
from flappy_env import FlappyEnv

LOG_DIR = "./flappy_logs/"
MODEL_DIR = "./flappy_models/"
BEST_MODEL_DIR = "./flappy_models/best_model/"
BEST_MODEL_PATH = os.path.join(BEST_MODEL_DIR, "best_model.zip")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

env = DummyVecEnv([lambda: FlappyEnv(render_mode=None)])

print("--- PREPARANDO TREINO FLAPPY BIRD ---")

if os.path.exists(BEST_MODEL_PATH):
    print(f"✅ Encontrei um modelo campeão salvo em: {BEST_MODEL_PATH}")
    print("🔄 Carregando para CONTINUAR o treinamento...")
    
    model = PPO.load(BEST_MODEL_PATH, env=env, tensorboard_log=LOG_DIR, print_system_info=True)
    
else:
    print("✨ Nenhum save encontrado. Criando cérebro NOVO do zero.")
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.0003)

eval_callback = EvalCallback(
    env, 
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR, 
    eval_freq=10000, 
    deterministic=True, 
    render=False,
    n_eval_episodes=5
)

print("🚀 Iniciando treinamento...")
print("Pressione CTRL+C para pausar (o melhor modelo já estará salvo).")

try:
    model.learn(total_timesteps=3000000, callback=eval_callback, reset_num_timesteps=False)
    print("✅ Treino finalizado com sucesso!")
except KeyboardInterrupt:
    print("\n⚠️ Treino interrompido pelo usuário.")
    print("O modelo 'best_model.zip' continua salvo e seguro.")

model.save(f"{MODEL_DIR}/flappy_final_backup")