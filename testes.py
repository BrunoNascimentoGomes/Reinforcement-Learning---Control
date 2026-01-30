import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.buffer import BufferMADDPG
from utils.MADDPGMod import MADDPG

# ---------------------------------------------------------
# 1. AMBIENTE SIMULADO SIMPLES (Navegação)
# ---------------------------------------------------------
class SimpleEnv:
    def __init__(self):
        self.num_agents = 2
        self.state_dim = 2   # Coordenadas (x, y)
        self.action_dim = 2  # Velocidade (vx, vy)
        self.max_action = 1.0
        
        # Alvos fixos para cada agente
        self.targets = np.array([
            [0.5, 0.5],   # Alvo do Agente 0
            [-0.5, -0.5]  # Alvo do Agente 1
        ])
        
        self.agents_pos = np.zeros((self.num_agents, 2))

    def reset(self):
        # Inicia agentes em posições aleatórias entre -1 e 1
        self.agents_pos = np.random.uniform(-1, 1, (self.num_agents, 2))
        return self.agents_pos.copy()

    def step(self, actions):
        # actions: shape (2, 2)
        # Atualiza posição (Simulando física simples: pos = pos + action * dt)
        self.agents_pos += actions * 0.1 
        
        # Clip para manter dentro do quadrado [-1, 1]
        self.agents_pos = np.clip(self.agents_pos, -1, 1)
        
        # Calcula Recompensa: Negativo da distância até o alvo
        rewards = []
        for i in range(self.num_agents):
            dist = np.linalg.norm(self.agents_pos[i] - self.targets[i])
            # Recompensa é -distância (quanto mais perto de 0, melhor)
            rewards.append(-dist)
            
        rewards = np.array(rewards).reshape(self.num_agents, 1)
        
        # Next state é a própria posição nova
        next_states = self.agents_pos.copy()
        
        # Done (não temos condição de parada fixa neste teste contínuo)
        done = False
        
        return next_states, rewards, done

# ---------------------------------------------------------
# 2. CONFIGURAÇÃO DO TREINAMENTO
# ---------------------------------------------------------
def run_test():
    # Hiperparâmetros
    NUM_EPISODES = 1000
    STEPS_PER_EPISODE = 50  # Episódios curtos para iterar rápido
    BATCH_SIZE = 64
    BUFFER_CAPACITY = 10000
    
    # Inicializa Ambiente
    env = SimpleEnv()
    
    # Inicializa Buffer (Versão Normal, SEM prioridade)
    buffer = BufferMADDPG(
        buffer_capacity=BUFFER_CAPACITY,
        batch_size=BATCH_SIZE,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        num_agents=env.num_agents
    )
    
    # Inicializa MADDPG
    maddpg = MADDPG(
        num_agents=env.num_agents,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        max_action=env.max_action,
        buffer=buffer,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    history_rewards = []
    
    print("Iniciando treinamento de teste...")

    # ---------------------------------------------------------
    # 3. LOOP PRINCIPAL
    # ---------------------------------------------------------
    for episode in range(NUM_EPISODES):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(STEPS_PER_EPISODE):
            # Selecionar Ação (com ruído para exploração)
            # Reduzimos o ruído conforme o treino avança
            noise = max(0.01, 1.0 - episode / 200)
            
            # Nota: select_action espera inputs e retorna arrays
            actions = maddpg.select_action(obs, noise=noise)
            
            # Passo no ambiente
            next_obs, rewards, done = env.step(actions)
            
            # Salvar no Buffer
            # Buffer espera tupla: (states, actions, rewards, next_states)
            # Shapes esperados pelo seu buffer.py:
            # States: (NumAgents, Dim)
            # Rewards: (NumAgents, 1)
            buffer.record((obs, actions, rewards, next_obs))
            
            # Treinar (só começa depois que o buffer tiver dados suficientes)
            if buffer.buffer_counter > BATCH_SIZE:
                maddpg.train()
            
            obs = next_obs
            episode_reward += np.sum(rewards) # Soma recompensa dos 2 agentes

        history_rewards.append(episode_reward / STEPS_PER_EPISODE)
        
        if episode % 20 == 0:
            print(f"Episódio {episode} | Recompensa Média: {history_rewards[-1]:.4f} | Ruído: {noise:.2f}")

    # ---------------------------------------------------------
    # 4. PLOTAR RESULTADOS
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(history_rewards)
    plt.title("Curva de Aprendizado MADDPG (Teste Simples)")
    plt.xlabel("Episódios")
    plt.ylabel("Recompensa Média (Neg. Distância)")
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # 5. TESTE DE INFERÊNCIA (SEM RUÍDO)
    # ---------------------------------------------------------
    print("\nTeste Final (Visualização de coordenadas):")
    obs = env.reset()
    for _ in range(10): # 5 passos
        actions = maddpg.select_action(obs, noise=0, deterministic=True)
        next_obs, rewards, _ = env.step(actions)
        print(f"Agente 0 Pos: {obs[0]} -> Alvo: {env.targets[0]}")
        print(f"Agente 1 Pos: {obs[1]} -> Alvo: {env.targets[1]}")
        print("-" * 30)
        obs = next_obs

if __name__ == "__main__":
    run_test()