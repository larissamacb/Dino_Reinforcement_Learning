import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# Importe a classe que você criou
from dino_env import DinoEnv 

# --- Configuração de Treinamento ---
# --- Configuração de Treinamento ---
LOG_DIR = "./dino_tensorboard/"
MODEL_SAVE_PATH = "./dino_dqn_checkpoints/"
TOTAL_TIMESTEPS = 5_000_000 # <-- MUDANÇA: 5 Milhões de passos
CHECKPOINT_FREQ = 100_000 

# Cria os diretórios se não existirem
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)


# --- 🚀 INÍCIO DA SELEÇÃO DE MODO ---
print("="*30)
print("🤖 ESCOLHA O MODO DE TREINAMENTO")
print("="*30)
print("   [1] Modo Lento (Visível, para espiar)")
print("   [2] Modo Rápido (Cego, para treinar)")
print("-"*30)
choice = input("Digite 1 ou 2 (padrão é 2): ")

render_mode_choice = None
if choice == "1":
    render_mode_choice = "human"
    print("\nIniciando em MODO LENTO (visível)...")
else:
    print("\nIniciando em MODO RÁPIDO (cego)...")
# --- FIM DA SELEÇÃO DE MODO ---


# 1. Crie o ambiente
env = DummyVecEnv([lambda: DinoEnv(render_mode=render_mode_choice)])


# --- 🚀 O CÉREBRO FINAL (INTELIGÊNCIA) ---

# Define a arquitetura do cérebro: [256 neurônios, 256 neurônios]
# (O padrão é [64, 64]. O anterior era [128, 128])
policy_kwargs = dict(net_arch=[256, 256])

# 2. Crie o "cérebro" (o agente)
model = DQN(
    "MlpPolicy", 
    env, 
    policy_kwargs=policy_kwargs, # <-- ADICIONA O CÉREBRO GIGANTE
    verbose=1, 
    tensorboard_log=LOG_DIR,
    learning_rate=0.0001,
    buffer_size=1000000,         # Memória (1 Milhão)
    learning_starts=100000,      # "Infância" longa (100k)
    batch_size=32,
    gamma=0.99,
    exploration_fraction=0.5,    # Explorar por 50% do tempo
    exploration_final_eps=0.01,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
)

# ... (O resto do seu train.py está perfeito) ...

# Callback para salvar checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ, 
    save_path=MODEL_SAVE_PATH, 
    name_prefix="dino_dqn_model"
)

print("--- O Treinamento vai começar ---")
print(f"Salvando logs em: {LOG_DIR}")
print(f"Salvando modelos em: {MODEL_SAVE_PATH}")
print("Pressione CTRL+C para parar o treino (o progresso será salvo no próximo checkpoint).")


# 3. Mande o cérebro aprender!
try:
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )
except KeyboardInterrupt:
    print("\nTreinamento interrompido pelo usuário.")

# 4. Salve o cérebro treinado final
model.save("dino_dqn_final")

print("--- Treinamento Concluído (ou interrompido) ---")
print(f"Modelo final salvo como 'dino_dqn_final.zip'")
env.close()