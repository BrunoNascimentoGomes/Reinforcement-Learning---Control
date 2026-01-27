import numpy as np
import torch
import random
from tanque.tanques import Quatro_Tanques
from utils.buffer import BufferMADDPG_Done
from utils.MADDPGMod import MADDPG_Done
import matplotlib.pyplot as plt
import math
from utils import pid



ACTION_DIM = 1     # Cada agente controla uma entrada (u1 ou u2)
MAX_ACTION = 1.0   # Exemplo: Limite a entrada de controle (Pois o processo recebe a ação normalizada)


MAX_STEPS = 1200
max_steps = MAX_STEPS


env = Quatro_Tanques(render_mode='human', max_steps=MAX_STEPS)

pid1 = pid.PID_Velo_Form(Kc=3.0, tau_i=30, u_min=0,u_max = 5, u_init= 2.4202)
pid2 = pid.PID_Velo_Form(Kc=2.7, tau_i=40, u_min=0,u_max = 5,u_init=1.6304)

         


print("\n" + "="*40)
print(" INICIANDO AVALIAÇÃO COM AÇÃO DETERMINÍSTICA")
print("="*40)

# Quantos episódios de teste você quer rodar
NUM_EVAL_EPISODES = 1

for eval_ep in range(NUM_EVAL_EPISODES):
    obs, info = env.reset()
    
    terminated = False
    truncated = False
    step_count = 0
    recompensa1 = 0
    recompensa2 = 0
    eval_reward = 0.0
    for i in range(MAX_STEPS):
        # AQUI ESTÁ A MUDANÇA: deterministic=True e noise=0.0

        error1 = obs[0][0]
        error2 = obs[1][0]

        error1 = env.denormalize(error1, 'e')
        error2 = env.denormalize(error2, 'e')

        action1 = pid1.compute_control(error1)
        action2 = pid2.compute_control(error2)

        actions = np.array([action1, action2])
        if i == 100:
            env.setpoint = [9.2,6.35]
        if i == 300:
            env.setpoint=[6.2,6.35]
        if i == 500:
            env.setpoint = [6.2, 4.35]
        if i == 700:
            env.setpoint=[6.2,6.35]
        obs, reward, terminated, truncated, info = env.step(u = actions)
        recompensa1 += reward[0]
        recompensa2 += reward[1]
        
        step_count += 1
    eval_reward = recompensa1 + recompensa2
    ISE1 = env.ISE1
    ISE2 = env.ISE2

    print(f"Avaliação {eval_ep+1}/{NUM_EVAL_EPISODES} | Steps: {step_count} | Recompensa Total: {eval_reward:.4f} | Terminated: {terminated}")

    # Renderiza o último episódio para você ver o gráfico de controle final
    if eval_ep == NUM_EVAL_EPISODES - 1:
        print("Gerando gráfico do último episódio de teste...")
        plt.figure(figsize=(10, 8)) # Garante uma nova figura
        env.render()
        #plt.show() # Bloqueia o script para mostrar a janela    
        print('ISE1 :', ISE1)
        print('ISE2 :', ISE2)
        plt.savefig('Controle_PID_Comp.png') # Salva em arquivo
        plt.show()                  # Mostra na tela