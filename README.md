# 🎮 Reinforcement Learning Arcade Agents

Este repositório contém implementações de agentes de Inteligência Artificial treinados para jogar clássicos do arcade utilizando **Aprendizado por Reforço (Reinforcement Learning)**.

O foco do projeto é a aplicação prática do algoritmo **PPO (Proximal Policy Optimization)** em ambientes customizados criados com a API Gymnasium, demonstrando como "sensores matemáticos" podem ser mais eficientes que o processamento de imagens para treinar IAs em jogos de física.

## 🧠 Sobre a Abordagem

Diferente de IAs que "olham" para a tela (pixels/CNNs), estes agentes tomam decisões baseadas em dados numéricos extraídos diretamente da memória do jogo (Feature Extraction). Isso torna o treino extremamente rápido e leve.

### Tecnologias Utilizadas
* **Python 3.x**
* **[Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3):** Implementação do algoritmo PPO.
* **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium):** Criação dos ambientes de treino.
* **[Pygame](https://www.pygame.org/):** Motor gráfico para a simulação dos jogos.

---

## 🕹️ Os Projetos

### 1. Chrome Dino Run AI 🦖
A IA aprende a controlar o famoso dinossauro do navegador, reagindo a cactos e pássaros.

* **Ambiente:** `dino_env.py`
* **O que a IA vê (Observação):** Um vetor de 4 valores normalizados:
    1.  **Posição Y** do Dinossauro.
    2.  **Tempo até o Impacto:** Cálculo físico (`distância / velocidade`) que permite à IA reagir independente da velocidade do jogo.
    3.  **Altura do Obstáculo:** Para decidir entre pular ou agachar.
    4.  **Tipo do Obstáculo:** Cacto (0.0) ou Pássaro (1.0).
* **Ações:** 3 (Correr, Pular, Agachar).
* **Estratégia de Treino:** Utiliza `ent_coef` (coeficiente de entropia) para incentivar a exploração inicial e evitar que a IA vicie na ação de "apenas correr".

### 2. Flappy Bird AI 🐦
A IA aprende a controlar o pássaro para passar pelos canos, lidando com gravidade e inércia.

* **Ambiente:** `Flappy_AI/flappy_env.py`
* **O que a IA vê (Observação):** Um vetor de 4 valores:
    1.  **Distância Horizontal** até o próximo cano.
    2.  **Distância Vertical** até o centro do vão do cano.
    3.  **Altura Atual** do pássaro.
    4.  **Velocidade Vertical:** Essencial para controlar o momento da queda/subida.
* **Ações:** 2 (Não fazer nada, Bater Asas).
* **Estratégia de Treino:** Utiliza um sistema de `EvalCallback` ("Modo Campeão") que salva separadamente o melhor modelo encontrado durante o treino para garantir performance máxima.

---

## 🌀 Como Rodar

### Pré-requisitos
Certifique-se de ter o Python instalado e as dependências do projeto:

```bash
pip install gymnasium stable-baselines3 pygame shimmy tensorboard tqdm rich
```

### Executando o Dino AI e o Flappy AI

Os arquivos do Dino estão na pasta `Dino_AI` e os arquivos do Flappy Bird estão na pasta `Flappy_AI`. Use cd para entrar na pasta correspondente conforme o exemplo:

```bash
cd Dino_AI
```

Se quiser sair da pasta para entrar em outra, use o comando abaixo antes de usar o comando acima novamente:

```bash
cd ..
```

**Para Treinar:**
```bash
python train.py
```

O terminal vai avisar quando o treinamento tiver acabado.

**Para Assistir (A IA jogando):**
```bash
python play.py
```

Se quiser usar os exemplos já treinados para ver a IA jogando perfeitamente sem precisar treinar no seu computador, arraste **todos** os arquivos da pasta `EXEMPLO DE TREINAMENTO` respectiva ao jogo que você deseja para a raiz **da pasta desse jogo**. Antes, certifique-se de apagar (ou deixar dentro de uma pasta temporária que você pode criar), para cada jogo, as seguintes pastas:

**Para o Dino_AI**
```bash
dino_dqn_checkpoints
dino_tensorboard
dino_dqn_final.zip
```

**Para o Flappy_AI**
```bash
flappy_logs
flappy_models
```

---

## 🌀 Demonstração

Abaixo estão vídeos resumidos (de aproximadamente 1 minuto cada) do progresso da IA para os dois jogos:

https://github.com/user-attachments/assets/7cf8f76b-884e-4c15-9d41-1f38fe542e50
https://github.com/user-attachments/assets/747be8d1-15d3-4246-b478-58fa02995f5b

---

✏️ Projeto desenvolvido por [Larissa](https://github.com/larissamacb) e [Samuel](https://github.com/SamuelGdA)
