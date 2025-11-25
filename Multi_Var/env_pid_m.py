# Usei os imports que você tinha no seu script original
from coluna_binary import DistilationColumn
from utils.pid import PID

import numpy as np

max_steps = 6500

# [MUDANÇA 1]: 
# Inicializando a planta sem o 'u=[0,0]'. 
# Isso faz com que ela use os valores padrão (u=[2.706, 3.706]),
# o que define self.Vr e self.Ls com valores válidos desde o início.

x = np.array([0.989996,   0.98506869, 0.97890509, 0.97123076, 0.96173051, 0.95005371,
 0.93582724, 0.91867871, 0.89827236, 0.87435801, 0.84683005, 0.81578814,
 0.78158594, 0.74485109, 0.70646163, 0.66747398, 0.6290119,  0.59213988,
 0.55775034, 0.52648847, 0.49872563, 0.47417117, 0.4455474,  0.41300484,
 0.37704334, 0.33853025, 0.29864485, 0.25874749, 0.22020013, 0.18418648,
 0.15157992, 0.12288666, 0.09826274, 0.07758201, 0.0605254,  0.04666715,
 0.03554402, 0.02670329, 0.01973121, 0.01426654, 0.010004])

env = DistilationColumn(max_steps=max_steps, state = x) 

#env = DistilationColumn(max_steps=max_steps) 

# [MUDANÇA 2]: 
# Aplicando os ganhos MAIS BAIXOS que sugeri para estabilizar o controle.
# (Kp=0.1, Ki=0.05) em vez de (Kp=1, Ki=0.5).
# Também defini limites de saturação (u_min=0) para segurança.
pid_d = PID(Kp=3.416, Ki=0.0455, Kd=0.0, dt=env.dt)
pid_b = PID(Kp=-2.74, Ki=-0.0365, Kd=0.0, dt=env.dt)



obs, _ = env.reset(state=x)

print(f"Iniciando simulação por {max_steps} passos...")
print(f"Estado Inicial Topo (xD): {env.x[0]}")
print(f"Estado Inicial Fundo (xB): {env.x[-1]}")

u_d_ss = 2.706
u_b_ss = 3.206

for i in range(max_steps):
    obs_d = obs[0]
    obs_b = obs[1]

    # Erros vêm da observação
    error_d = obs_d[0]
    error_b = obs_b[0]
    
    # Calcula a ação de controle (saturação já ocorre dentro do PID)
    u_d = pid_d.compute(error_d) +u_d_ss # Ação para o Topo (Lr - Refluxo)
    u_b = pid_b.compute(error_b) +u_b_ss# Ação para o Fundo (Vs - Vaporização)
    
    #print(u_d)

    
    if i == 100:
        env.setpoint = [0.996, 0.01]
        #A
        '''
        u_d += 0.01
        u_b -= 0
        '''
    # Ações [Lr, Vs]
    u = [u_d, u_b] 

    # Envia as ações para o ambiente
    next_obs, reward, terminated, truncated, info = env.step(u)
    obs = next_obs
    

    if (i+1) % 5000 == 0:
         pass
         #print(f"Passo {i+1}/{max_steps}...")
         #env.plot_estagio()
         #env.plot_u()
         
    if terminated or truncated:
        break

print("Simulação concluída.")
print("Gerando gráficos...")
env.render()

print("Execução finalizada.")
