import gymnasium as gym
from gymnasium import spaces
import numpy as np
import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray import tune
import os

# Importa sua planta original
from modelos import ModeloSegundaOrdem

# ==============================================================================
# 1. ADAPTADOR DO AMBIENTE (Wrapper)
# O Ray precisa que a classe herde de gymnasium.Env e tenha 'action_space' definido
# ==============================================================================
class PlantEnvWrapper(gym.Env):
    def __init__(self, env_config):
        # Configurações vindas do dicionário env_config
        K = env_config.get("K", 1.0)
        wn = env_config.get("wn", 2.0)
        zeta = env_config.get("zeta", 0.3)
        dt = env_config.get("dt", 0.02)
        
        # Instancia sua planta original
        self.plant = ModeloSegundaOrdem(K=K, wn=wn, zeta=zeta, dt=dt, max_steps=500)
        
        # Define os espaços de ação e observação (Obrigatório para o Ray)
        # Ação: contínua entre -1 e 1
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observação: 4 estados [erro, y, du, sp] (normalizados entre -1 e 1)
        # Colocamos limites um pouco maiores (-2 a 2) por segurança, ou -inf a inf
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # O Ray exige que o reset retorne (obs, info)
        obs, info = self.plant.reset()
        # O Ray às vezes pede setpoint aleatório no reset, sua planta já faz isso?
        # Se não, podemos forçar aqui:
        self.plant.setpoint = np.random.uniform(1.0, 5.0)
        return obs, info

    def step(self, action):
        # O Ray envia a ação como um array numpy
        obs, reward, terminated, truncated, info = self.plant.step(action)
        return obs, reward, terminated, truncated, info

    def render(self):
        self.plant.render()

# ==============================================================================
# 2. CONFIGURAÇÃO E TREINAMENTO COM RAY RLLIB
# ==============================================================================
if __name__ == "__main__":
    # Inicializa o Ray
    ray.init(ignore_reinit_error=True)

    # Registra o ambiente personalizado no Ray
    tune.register_env("minha_planta_v0", lambda config: PlantEnvWrapper(config))

    # Configuração do Algoritmo DDPG
    config = (
        DDPGConfig()
        .environment(
            env="minha_planta_v0",
            env_config={
                "K": 1.0, "wn": 2.0, "zeta": 0.3, "dt": 0.05
            }
        )
        .framework("torch")  # Usa PyTorch
        .training(
            gamma=0.99,
            lr=1e-3,                # Learning rate do crítico e ator
            tau=0.005,              # Soft update
            train_batch_size=64,    # Mesmo batch que discutimos
            # Configuração da Rede Neural (Ator e Crítico)
            model={
                "fcnet_hiddens": [400, 300],
                "fcnet_activation": "relu",
            },
            # Replay Buffer
            replay_buffer_config={
                "type": "MultiAgentPrioritizedReplayBuffer",
                "capacity": 50000,
            },
            # Configuração de Ruído (Exploração)
            exploration_config={
                "type": "OrnsteinUhlenbeckNoise",
                "scale_timesteps": 10000, # Decaimento do ruído ao longo dos steps
                "initial_scale": 1.0,
                "final_scale": 0.02,
                "ou_base_scale": 0.1,
                "ou_theta": 0.15,
                "ou_sigma": 0.2,
            }
        )
        .resources(num_gpus=0) # Mude para 1 se tiver GPU configurada
        .rollouts(num_rollout_workers=1) # Paralelismo: 1 worker coletando dados
    )

    # Constrói o algoritmo
    algo = config.build()

    print("--- Iniciando Treinamento com Ray RLlib ---")
    
    # Loop de Treino
    for i in range(50): # Treina por 50 iterações (cada iteração são vários episódios)
        result = algo.train()
        
        print(f"Iteração: {i+1}")
        print(f"  Reward Médio: {result['env_runners']['episode_reward_mean']:.2f}")
        print(f"  Min Reward:   {result['env_runners']['episode_reward_min']:.2f}")
        print(f"  Len Médio:    {result['env_runners']['episode_len_mean']:.2f}")
        print("-" * 30)

    # ==========================================================================
    # 3. TESTE / INFERÊNCIA
    # ==========================================================================
    print("\n--- Testando o Agente Treinado ---")
    env = PlantEnvWrapper({})
    obs, _ = env.reset()
    env.plant.setpoint = 1.0 # Força setpoint fixo para teste visual
    
    total_reward = 0
    done = False
    
    while not done:
        # Pede ação para o agente treinado (explore=False desliga o ruído)
        action = algo.compute_single_action(obs, explore=False)
        
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    print(f"Recompensa Total no Teste: {total_reward:.2f}")
    env.render() # Usa o render da sua classe original
    
    # Finaliza
    ray.shutdown()