import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class TITOSystem:
    def __init__(self, K, wn1, wn2, zeta1, zeta2, dt=0.1, max_steps=1000,
                setpoint=[0, 0], render_mode=None, tol=1e-3):
        
        # ... (Seu código de inicialização anterior mantém igual) ...
        self.state = np.array([0, 0, 0, 0], dtype=float)
        self.K = K
        self.wn1 = wn1
        self.wn2 = wn2
        self.zeta1 = zeta1
        self.zeta2 = zeta2   
        self.setpoint = setpoint
        self.dt = dt
        self.t = 0.0
        self.num_step = 0
        self.tol = tol
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        

        # Históricos...
        self.y1_hist = []
        self.y2_hist = []
        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []

        self.last_u = np.array([0.0, 0.0])

       
        self.norm_limits = {
            'y':  (0.0, 10.0),      # Saída
            'dy': (-5.0,5),
            'sp': (0.0, 10.0),      # Setpoint
            'u':  (-6.0, 6.0),      # Ação absoluta
            'e':  (-10.0, 10.0),    # Erro
            'du': (-12.0, 12.0),       # Variação da ação (Delta U)
            'int_e':(-20.0,20.0)
        }

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

    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 0.0
        self.num_step = 0

        self.last_u = np.array([0.0, 0.0])

        # Limpa históricos...
        self.y1_hist = []
        self.y2_hist = []
        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []

        # Estado Inicial Físico
        y1, y2 = 0.0, 0.0
        u1_prev, u2_prev = 0.0, 0.0
        dy1 = 0
        dy2 = 0
        self.int_err1 = 0
        self.int_err2 = 0   

        err1 = self.setpoint[0] - y1
        err2 = self.setpoint[1] - y2

        u_prev1, u_prev2 = 0.0, 0.0
        du1, du2 = 0.0, 0.0

        # --- NORMALIZAÇÃO NA OBSERVAÇÃO ---
        # Obs: [norm_error, norm_y, norm_u_prev, norm_sp]
        # --- OBSERVAÇÃO COM 5 VARIÁVEIS ---
        # Obs: [erro, y, u_atual, delta_u, setpoint]
        obs1 = np.array([
            self._normalize(err1, 'e'),
            self._normalize(self.int_err1,'int_e'),
            self._normalize(y1, 'y'),
            self._normalize(dy1, 'dy'),
            self._normalize(u_prev1, 'u'),  # Informação Absoluta
            self._normalize(du1, 'du'),     # Informação de Variação
            self._normalize(self.setpoint[0], 'sp')
        ], dtype=np.float32)

        obs2 = np.array([
            self._normalize(err2, 'e'),
            self._normalize(self.int_err2,'int_e'),
            self._normalize(y2, 'y'),
            self._normalize(dy2, 'dy'),
            self._normalize(u_prev2, 'u'),
            self._normalize(du2, 'du'),
            self._normalize(self.setpoint[1], 'sp')
        ], dtype=np.float32)

        return np.array([obs1, obs2]), {}

    # ... (Método _ode continua igual) ...
    def _ode(self, t, x, u):
        x1, x2, x3, x4 = x
        u1, u2 = u
        K11, K12 = self.K[0][0], self.K[0][1]
        K21, K22 = self.K[1][0], self.K[1][1]

        dx1dt = x2
        dx2dt = -2*self.zeta1*self.wn1*x2 - (self.wn1**2)*x1 + (self.wn1**2)*(K11*u1 + K12*u2)
        dx3dt = x4
        dx4dt = -2*self.zeta2*self.wn2*x4 - (self.wn2**2)*x3 + (self.wn2**2)*(K21*u1 + K22*u2)
        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def step(self, actions_norm):
        # 1. RECEBE AÇÃO NORMALIZADA DO AGENTE [-1, 1] E CONVERTE PARA FÍSICA
        u_phys = self._denormalize_action(np.array(actions_norm).reshape(-1))
        
        delta_u = u_phys - self.last_u
        
        self.last_u = u_phys
        # 2. SIMULAÇÃO FÍSICA (Solve IVP usa valores reais)
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
        y2 = float(self.state[2])

        dy1 = float(self.state[1])
        dy2 = float(self.state[3])

        
        # Históricos (salvamos o valor FÍSICO para plotar bonito)
        self.num_hist.append(self.num_step)
        self.y1_hist.append(y1)
        self.y2_hist.append(y2)
        self.u1_hist.append(u_phys[0])
        self.u2_hist.append(u_phys[1])
        self.setpoint1_hist.append(self.setpoint[0])
        self.setpoint2_hist.append(self.setpoint[1])

        error1 = self.setpoint[0] - y1
        error2 = self.setpoint[1] - y2


        self.int_err1 += error1 * self.dt
        self.int_err1 = np.clip(self.int_err1, -20.0, 20.0)
        
        self.int_err2 += error2 * self.dt
        self.int_err2 = np.clip(self.int_err2, -20.0, 20.0)

        # 4. OBSERVAÇÃO COM 5 VARIÁVEIS
        #Obs: [erro, integral_erro, y, u, delta_u, setpoint]
        obs1 = np.array([
            self._normalize(error1, 'e'),
            self._normalize(self.int_err1,'int_e'),
            self._normalize(y1, 'y'),
            self._normalize(dy1, 'dy'),
            self._normalize(u_phys[0], 'u'), # U Absoluto
            self._normalize(delta_u[0], 'du'), # Delta U
            self._normalize(self.setpoint[0], 'sp')
        ], dtype=np.float32)

        obs2 = np.array([
            self._normalize(error2, 'e'),
            self._normalize(self.int_err2,'int_e'),
            self._normalize(y2, 'y'),
            self._normalize(dy2, 'dy'),
            self._normalize(u_phys[1], 'u'), # U Absoluto
            self._normalize(delta_u[1], 'du'), # Delta U
            self._normalize(self.setpoint[1], 'sp')
        ], dtype=np.float32)

        obs = np.array([obs1, obs2])

        #Original: -10 e -1
        r_error1 = -10.0 * (error1**2) if abs(error1) > 0.05 else 0
        bonus1 = 2 if abs(error1) < 0.05 else 0
        
        r_smooth1 = -10.0 * (delta_u[0]**2)
        r_effort1 = -0.05 * (u_phys[0]**2)
        
        r_error2 = -10.0 * (error2**2) if abs(error2) > 0.05 else 0
        bonus2 = 0 if abs(error2) < 0.05 else 0
        
        r_smooth2 = -10.0 * (delta_u[1]**2)
        r_effort2 = -0.05 * (u_phys[1]**2)

        reward1 = r_error1 + r_smooth1 + r_effort1 + bonus1
        reward2 = r_error2 + r_smooth2 + r_effort2+bonus2
        
        # Bônus de precisão
        '''
        if abs(error1) < 0.05: reward1 += 1.0
        if abs(error2) < 0.05: reward2 += 1.0
        '''
        if abs(error1) < 0.005: reward1 += 5.0
        if abs(error2) < 0.005: reward2 += 5.0

        reward = np.array([reward1, reward2])

        terminated = False
        truncated = self.num_step >= self.max_steps
        info = {}

        return obs, reward, terminated, truncated, info
    
    # ... (render continua igual) ...
    def render(self):
        if self.render_mode is None: return
        plt.clf()
        # ... (código de plotagem usando self.y1_hist, etc - que já são físicos) ...
        # (Use o código de render que você já tem, ele vai plotar os dados físicos corretamente)
        plt.subplot(2, 2, 1)
        plt.plot(self.num_hist, self.y1_hist, label='Output y1')
        plt.plot(self.num_hist, self.setpoint1_hist, 'r--', label='Setpoint 1')
        plt.legend(); plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(self.num_hist, self.y2_hist, label='Output y2')
        plt.plot(self.num_hist, self.setpoint2_hist, 'r--', label='Setpoint 2')
        plt.legend(); plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(self.num_hist, self.u1_hist, label='Input u1', color='g')
        plt.legend(); plt.grid(True)
        
        plt.subplot(2, 2, 4)
        plt.plot(self.num_hist, self.u2_hist, label='Input u2', color='g')
        plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)