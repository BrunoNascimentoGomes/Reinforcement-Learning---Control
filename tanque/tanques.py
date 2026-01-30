import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import math

class Quatro_Tanques:
    def __init__(self, dt=4, render_mode=None, setpoint=[6.2,6.35], max_steps=500, tol=0.005):
        
        self.dt = dt
        self.state = np.array([6.2,6.35,0.4836,1.1440], dtype = float)
        self.u_inicial = [2.4202, 1.6304]
        self.last_u = [2.4202, 1.6304]

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

        
        #Fase Mínima
        self.k1 = 3.33 #cm2/Vs
        self.k2 = 3.35 #cm2/Vs

        self.gamma1 = 0.7
        self.gamma2 = 0.6
        

        '''
        #Fase Não-Mínima:
        self.k1 = 3.24 #cm2/Vs
        self.k2 = 3.29 #cm2/Vs

        self.gamma1 = 0.43
        self.gamma2 = 0.34
        '''
        
        self.g = 981 #cm2/s2
        
        
        #Plots
        self.h1_hist = []
        self.h2_hist = []
        self.h3_hist = []
        self.h4_hist = []

        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []

        self.norm_limits = {
            'y':(0.0,12),
            'u':(0.0 ,5.0),
            'e':(-10,10),
            'sp':(0.0,10),
            'deltau':(-5,5),
            'h3':(0,3),
            'h4':(0,4)
        }

        
        #Análise de Malha:
        self.ISE1 = 0.0
        self.ISE2 = 0.0
        self.SSMV1 = 0.0
        self.SSMV2 = 0.0

        
    # Função auxiliar para normalizar [-1, 1]
    def _normalize(self, value, key):
        min_v, max_v = self.norm_limits[key]
        # Clipa para garantir que não passe dos limites e estrague a rede
        value = np.clip(value, min_v, max_v)
        return 2 * (value - min_v) / (max_v - min_v) - 1

    # Função auxiliar para desnormalizar (Ação do Agente -> Física)
    def _denormalize_action(self, action_norm):
        # O agente entrega [-1, 1], convertemos para [u_min, u_max]
        min_v, max_v = self.norm_limits['u']
        # Fórmula inversa
        action_phys = 0.5 * (action_norm + 1) * (max_v - min_v) + min_v
        return action_phys

    def denormalize(self, value_norm, key):
        min_v, max_v = self.norm_limits[key]
        value_phys = 0.5 * (value_norm + 1) * (max_v - min_v) + min_v
        return value_phys
    
    def reset(self):

        self.state = np.array([6.2,6.35,0.4836,1.1440], dtype = float)
        self.setpoint=[6.2,6.35]
        self.u_inicial = [2.4202, 1.6304]
        self.last_u = [2.4202, 1.6304]
        self.t = 0
        self.num_step = 0

        self.y1_hist = []
        self.y2_hist = []


        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []

        y1 = self.state[0]
        y2 = self.state[1]
        h3 = self.state[2]
        h4 = self.state[3]

        err1 = self.setpoint[0] - y1
        err2 = self.setpoint[1] - y2

        delta_u = [0,0]

        obs1 = np.array([
            self._normalize(err1, 'e'),
            self._normalize(y1, 'y'),
            self._normalize(self.setpoint[0], 'sp'),
            self._normalize(delta_u[0], 'deltau'),
            self._normalize(h3, 'h3'),
            self._normalize(h4, 'h4')
        ], dtype=np.float32)

        obs2 = np.array([
            self._normalize(err2, 'e'),
            self._normalize(y2, 'y'),
            self._normalize(self.setpoint[1], 'sp'),
            self._normalize(delta_u[1], 'deltau'),
            self._normalize(h3, 'h3'),
            self._normalize(h4, 'h4')
        ], dtype=np.float32)

        #Análise de Malha:
        self.ISE1 = 0.0
        self.ISE2 = 0.0
        self.SSMV1 = 0.0
        self.SSMV2 = 0.0

        return np.array([obs1, obs2]), {}
    
    def _ode(self, t, state, u):
        h1, h2, h3, h4 = state
        u1, u2 = u
        h1 = max(h1, 0.0)
        h2 = max(h2, 0.0)
        h3 = max(h3, 0.0)
        h4 = max(h4, 0.0)
        dh1 = -(self.a1 / self.A1)*np.sqrt(2*self.g*h1) + (self.a3/self.A1)*np.sqrt(2*self.g*h3) + ((self.gamma1*self.k1)/self.A1)*u1
        dh2 = -(self.a2 / self.A2)*np.sqrt(2*self.g*h2) + (self.a4/self.A2)*np.sqrt(2*self.g*h4) + ((self.gamma2*self.k2)/self.A2)*u2
        dh3 = -(self.a3 / self.A3)*np.sqrt(2*self.g*h3) + (((1-self.gamma2)*self.k2)/self.A3) * u2
        dh4 = -(self.a4 / self.A4)*np.sqrt(2*self.g*h4) + (((1-self.gamma1)*self.k1)/self.A4) * u1

        return [dh1, dh2, dh3, dh4]


    def step(self, actions_norm = [0,0], u=None):
        u_phys = self._denormalize_action(np.array(actions_norm).reshape(-1))

        u_phys = np.clip(u_phys, 0.0, 5.0)

        u_phys = u_phys if (u is None) else u

        delta_u = u_phys - self.last_u

        self.last_u = u_phys

        self.SSMV1 += (delta_u[0]**2) *self.dt
        self.SSMV2 += (delta_u[1]**2) *self.dt

        sol = solve_ivp(
            self._ode,
            [self.t, self.t + self.dt],
            self.state,
            args=(u_phys,),
            method='RK45',
            t_eval=[self.t + self.dt],
            rtol=1e-4, atol=1e-6
        )

        self.state = sol.y[:, -1]
        self.t += self.dt
        self.num_step += 1

        y1 = float(self.state[0])
        y2 = float(self.state[1])

        h3 = float(self.state[2])
        h4 = float(self.state[3])
        

        self.num_hist.append(self.num_step)
        self.y1_hist.append(y1)
        self.y2_hist.append(y2)
        self.u1_hist.append(u_phys[0])
        self.u2_hist.append(u_phys[1])
        self.setpoint1_hist.append(self.setpoint[0])
        self.setpoint2_hist.append(self.setpoint[1])
        
        error1 = self.setpoint[0] - y1
        error2 = self.setpoint[1] - y2

        self.ISE1 += error1**2 *self.dt
        self.ISE2 += error2**2 *self.dt

        obs1 = np.array([
            self._normalize(error1, 'e'),
            self._normalize(y1, 'y'),
            self._normalize(self.setpoint[0], 'sp'),
            self._normalize(delta_u[0], 'deltau'),
            self._normalize(h3, 'h3'),
            self._normalize(h4, 'h4')
        ], dtype=np.float32)

        obs2 = np.array([
            self._normalize(error2, 'e'),
            self._normalize(y2, 'y'),
            self._normalize(self.setpoint[1], 'sp'),
            self._normalize(delta_u[1], 'deltau'),
            self._normalize(h3, 'h3'),
            self._normalize(h4, 'h4')
        ], dtype=np.float32) 

        obs = np.array([obs1, obs2])
        
        fator_erro = 1
        fator_u = 0.5


        error1_norm = self._normalize(error1,'e')
        error2_norm = self._normalize(error2,'e')
        deltaU1_norm = self._normalize(delta_u[0],'deltau')
        deltaU2_norm = self._normalize(delta_u[1],'deltau')

        '''
        reward1 = -fator_erro*(error1**2) - fator_u*(delta_u[0]**2)
        reward2 = -fator_erro*(error2**2) - fator_u*(delta_u[1]**2)
        '''

        reward1 = -fator_erro*(error1_norm**2) - fator_u*(deltaU1_norm**2)
        reward2 = -fator_erro*(error2_norm**2) - fator_u*(deltaU2_norm**2)
        
        c = 1
        k = 0.5
        if abs(error1) < self.tol:
            reward1 += +c * math.exp(-k*abs(error1_norm)) # Bônus extra por atingir o alvo
        
        if abs(error2) < self.tol:
            reward2 += +c * math.exp(-k*abs(error2_norm)) # Bônus extra por atingir o alvo
        
        terminated = [False, False]

        if y1 > 20 or y2 > 20 or y1 < 0 or y2 < 0:
            terminated = [True, True]
            #reward1 += -20
            #reward2 += -20
        
        
        truncated = self.num_step >= self.max_steps
        '''
        if truncated:
            reward1 += 1000
            reward2 += 1000
        '''
        reward = np.array([reward1, reward2])
        

        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode is None: return
        plt.clf()
        # ... (código de plotagem usando self.y1_hist, etc - que já são físicos) ...
        # (Use o código de render que você já tem, ele vai plotar os dados físicos corretamente)
        plt.subplot(2, 2, 1)
        plt.plot(self.num_hist, self.y1_hist, label='Output y1')
        plt.plot(self.num_hist, self.setpoint1_hist, 'r--', label='Setpoint 1')
        plt.legend(); plt.grid(True)
        plt.title('ISE: {:.2f} | SSMV: {:.2f}'.format(self.ISE1, self.SSMV1))


        
        plt.subplot(2, 2, 2)
        plt.plot(self.num_hist, self.y2_hist, label='Output y2')
        plt.plot(self.num_hist, self.setpoint2_hist, 'r--', label='Setpoint 2')
        plt.legend(); plt.grid(True)
        plt.title('ISE: {:.2f} | SSMV: {:.2f}'.format(self.ISE2, self.SSMV2))



        plt.subplot(2, 2, 3)
        plt.plot(self.num_hist, self.u1_hist, label='Input u1', color='g')
        plt.legend(); plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.num_hist, self.u2_hist, label='Input u2', color='g')
        plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        #plt.show()

if __name__ == "__main__":
    
    max_steps = 1000
    
    env = Quatro_Tanques(render_mode='human', max_steps=max_steps)
# Reinicia o ambiente
    obs, _ = env.reset()

# Sinal de controle constante para simplificar
    u = env.u_inicial

# Loop de simulação
    for step in range(max_steps):

    # Executa um passo no modelo
        obs, reward, terminated, truncated, info = env.step(u=u)

# Mostra o resultado (deve exibir a curva y e o setpoint variando)
    env.render()