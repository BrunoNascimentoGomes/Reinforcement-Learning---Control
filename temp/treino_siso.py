import numpy as np
import torch
import matplotlib.pyplot as plt
import os

# Importe suas classes corrigidas
from utils.DDPGAgent import DDPGAgent_Done
from modelo_siso import ModeloPrimeiraOrdem # Certifique-se que o arquivo tem a planta corrigida (instável)

# Configurações de Diretório
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')

# =============================
# Hiperparâmetros
# =============================
STATE_DIM = 3     # [y, setpoint, error]
ACTION_DIM = 1    # [u]
MAX_ACTION = 10.0  # Limite do atuador (importante para não saturar instantaneamente)
MAX_STEPS = 500   # Passos por episódio
N_EPISODES = 100  # Total de episódios 5000

# Configuração do Agente
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Treinando em: {device}")

agent = DDPGAgent_Done(
    state_dim=STATE_DIM,
    action_dim=ACTION_DIM,
    max_action=MAX_ACTION,
    buffer_capacity=50000,
    batch_size=128,      # Batch maior ajuda na estabilidade
    actor_lr=1e-4,       # LR conservador para planta instável
    critic_lr=1e-3,
    gamma=0.99,
    tau=0.005,
    device=device
)

# Ruído Exploratório (Ornstein-Uhlenbeck ou Gaussiano Decrescente)
noise_std = 2.0
noise_decay = 0.995
min_noise = 0.05

# Métricas
scores = []
avg_scores = []

# =============================
# Loop de Treino
# =============================
for ep in range(N_EPISODES):
    # Setpoint pode variar para generalização ou fixo para facilitar início
    # O artigo usa range [0, 10] ou similar. Vamos começar com 1.0 fixo.
    sp = 1.0 # np.random.uniform(0.5, 2.0) 
    
    # IMPORTANTE: dt pequeno (0.01 ou 0.05) é crucial para a dinâmica rápida da exponencial
    env = ModeloPrimeiraOrdem(dt=0.02, setpoint=sp, max_steps=MAX_STEPS)
    
    obs, _ = env.reset()
    score = 0
    done = False
    
    step_count = 0
    
    while not done:
        # 1. Selecionar Ação com Ruído
        state_tensor = torch.FloatTensor(obs)
        action = agent.select_action(state_tensor, noise=noise_std)
        
        # 2. Executar na Planta
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Lógica de Done
        done = terminated or truncated
        done_bool = float(terminated)
        
        # 3. Salvar na Memória (IMPORTANTE: Passar float(terminated))
        # terminated = True se explodiu (y > 100). Isso é um estado terminal real.
        # truncated = True se acabou o tempo. Matematicamente não é terminal, 
        # mas na prática costuma-se tratar igual ou diferenciar no buffer.
        # Para simplificar: use `terminated` como o sinal de done matemático.
         # 1.0 se falhou, 0.0 se continua ou acabou tempo
        
        agent.replay_buffer.record((obs, action, reward, next_obs, done_bool))
        
        # 4. Treinar
        if agent.replay_buffer.buffer_counter >= agent.batch_size:
            agent.train()
            
        obs = next_obs
        score += reward
        step_count += 1
        done = terminated or truncated
        if done:
            break
            
    # Atualizar ruído
    noise_std = max(min_noise, noise_std * noise_decay)
    
    # Logs
    scores.append(score)
    avg_score = np.mean(scores[-100:])
    avg_scores.append(avg_score)
    
    final_y = obs[0] # obs é [y, setpoint] conforme sua classe corrigida
    
    print(f"Ep {ep+1:3d} | Score: {score:8.2f} | Steps: {step_count:3d} | Noise: {noise_std:.2f}")

    # Salvar o melhor modelo
    if ep > 20 and score == max(scores):
        agent.save("checkpoints/best_model")

# =============================
# Visualização
# =============================
plt.figure(figsize=(10, 5))
plt.plot(scores, label='Score por Episódio')
plt.plot(avg_scores, label='Média Móvel (100 eps)')
plt.xlabel('Episódio')
plt.ylabel('Recompensa Total')
plt.title('Treinamento DDPG - Planta Instável')
plt.legend()
plt.savefig('learning_curve.png')
plt.show()


print("\nIniciando episódio de teste (Determinístico)...")

obs, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(obs)
    
    # IMPORTANTE: noise=0.0 e deterministic=True
    # Isso garante que estamos avaliando a "política pura" aprendida
    action = agent.select_action(state_tensor, noise=0.0, deterministic=True)
    
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    done = terminated or truncated
    obs = next_obs
    total_reward += reward
    

print(f"Episódio finalizado.")
print(f"Recompensa Total: {total_reward:.2f}")
print(f"Valor Final de y: {env.y_hist[-1]:.4f}")

# =============================
# Plotagem (Usando o render da sua classe)
# =============================
# Se quiser customizar o gráfico, pode acessar env.y_hist, env.u_hist, etc.
env.render()

print("Treinamento finalizado.")