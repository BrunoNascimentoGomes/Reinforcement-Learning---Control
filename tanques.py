import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Quatro_Tanques:
    def __init__(self, dt, render_mode, setpoint, max_steps, tol):
        
        self.dt = dt
        self.state = np.array([0,0,0,0], dtype = float)
        self.setpoint = setpoint # Array com 2 valores
        self.t = 0
        self.num_step = 0
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.tol = float(tol)



        #Parâmetros físicos
        self.A1 = 28 #cm2
        self.A2 = 32 #cm2
        self.A3 = 28 #cm2
        self.A4 = 32 #cm2

        self.a1 = 0.071 #cm2
        self.a2 = 0.051 #cm2
        self.a3 = 0.071 #cm2 
        self.a4 = 0.051 #cm2

        self.k1 = 3.33 #cm2/Vs
        self.k2 = 3.35 #cm2/Vs

        self.gamma1 = 0.7
        self.gamma2 = 0.6

        self.g = 981 #cm2/s2

        
        #Plots
        self.h1_hist = []
        self.h2_hist = []
        self.h3_hist = []
        self.h4_hist = []

        self.u1_hist = []
        self.u2_hist = []

        self.norm_limits = {
            #Preencher
        }
    # Função auxiliar para normalizar [-1, 1]
    def _normalize(self, value, key):
        min_v, max_v = self.norm_limits[key]
        # Clipa para garantir que não passe dos limites e estrague a rede
        return 2 * (value - min_v) / (max_v - min_v) - 1

    # Função auxiliar para desnormalizar (Ação do Agente -> Física)
    def _denormalize_action(self, action_norm):
        # O agente entrega [-1, 1], convertemos para [u_min, u_max]
        min_v, max_v = self.norm_limits['u']
        # Fórmula inversa
        action_phys = 0.5 * (action_norm + 1) * (max_v - min_v) + min_v
        return action_phys

    def reset(self):

        ...