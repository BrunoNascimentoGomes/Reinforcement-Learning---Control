import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class TITOSystem:
    def __init__(self, K, wn1, wn2, zeta1, zeta2, dt=0.1, max_steps=1000,
                setpoint = [0 ,0],render_mode=None, tol=1e-3):
        # Parâmetros do sistema
        self.state = np.array([0, 0, 0, 0], dtype = float) # Estado inicial [x1, x2, x3, x4]

        self.K = K
        self.wn1 = wn1
        self.wn2 = wn2
        self.zeta1 = zeta1
        self.zeta2 = zeta2   
        self.setpoint = setpoint


        self.dt = 0.1   # Passo de tempo para simulação discreta

        self.t = 0.0
        self.num_step = 0

        self.tol = tol
        self.max_steps = int(max_steps)
        self.render_mode = render_mode

        # Histórico para plotagem
        self.y1_hist = []
        self.y2_hist = []
        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []
    
    def reset(self):
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
        self.t = 0.0
        self.num_step = 0

        self.y1_hist = []
        self.y2_hist = []
        self.u1_hist = []
        self.u2_hist = []
        self.setpoint1_hist = []
        self.setpoint2_hist = []
        self.num_hist = []

        #obs = error, y, u[-1], self.setpoint
        obs1 = np.array([self.setpoint[0] - 0.0, 0.0, 0.0, self.setpoint[0]], dtype=np.float32)
        obs2 = np.array([self.setpoint[1] - 0.0, 0.0, 0.0, self.setpoint[1]], dtype=np.float32)

        obs  = np.array([obs1, obs2])
        return obs, {}
    
    def _ode(self, t, x, u):
        x1, x2, x3, x4 = x
        u1, u2 = u
        K11 = self.K[0][0]
        K12 = self.K[0][1]
        K21 = self.K[1][0]
        K22 = self.K[1][1]

        dx1dt = x2
        dx2dt = -2*self.zeta1*self.wn1*x2 - (self.wn1**2)*x1 + (self.wn1**2)*(K11*u1 + K12*u2)

        dx3dt = x4
        dx4dt = -2*self.zeta2*self.wn2*x4 - (self.wn2**2)*x3 + (self.wn2**2)*(K21*u1 + K22*u2)

        return [dx1dt, dx2dt, dx3dt, dx4dt]

    def step(self, u):
        u = np.asarray(u, dtype=float)

        # integração Euler rápida
        dx = self._ode(self.t, self.state, u)
        self.state += self.dt * np.array(dx)

        self.t += self.dt
        self.num_step += 1

        y1 = self.state[0]
        y2 = self.state[2]

        error1 = self.setpoint[0] - y1
        error2 = self.setpoint[1] - y2

        # observação rápida (sem criar arrays novos)
        obs = np.zeros((2,4), dtype=np.float32)
        obs[0] = [error1, y1, u[0], self.setpoint[0]]
        obs[1] = [error2, y2, u[1], self.setpoint[1]]

        # recompensa
        reward = np.array([
            -error1**2 if abs(error1) > self.tol else 10,
            -error2**2 if abs(error2) > self.tol else 10
        ], dtype=np.float32)

        terminated = False
        truncated = self.num_step >= self.max_steps

        return obs, reward, terminated, truncated, {}


    def render(self):
        if self.render_mode is None:
            return
            
            # Ajusta o tamanho da figura para acomodar 4 gráficos (grade 2x2)
        plt.figure(figsize=(15, 10))

            # --- Gráfico 1: Saída Y1 ---
        plt.subplot(2, 2, 1)
        plt.plot(self.num_hist, self.y1_hist, label='Output y1')
        plt.plot(self.num_hist, self.setpoint1_hist, 'r--', label='Setpoint 1')
        plt.xlabel('Time Steps')
        plt.ylabel('Output y1')
        plt.title('System Output y1 vs Setpoint')
        plt.legend()
        plt.grid(True)

            # --- Gráfico 2: Entrada U1 ---
        plt.subplot(2, 2, 3)
        plt.plot(self.num_hist, self.u1_hist, label='Input u1', color='g')
        plt.xlabel('Time Steps')
        plt.ylabel('Input u1')
        plt.title('Control Input u1')
        plt.legend()
        plt.grid(True)

            # --- Gráfico 3: Saída Y2 ---
        plt.subplot(2, 2, 2)
        plt.plot(self.num_hist, self.y2_hist, label='Output y2')
        plt.plot(self.num_hist, self.setpoint2_hist, 'r--', label='Setpoint 2')
        plt.xlabel('Time Steps')
        plt.ylabel('Output y2')
        plt.title('System Output y2 vs Setpoint')
        plt.legend()
        plt.grid(True)

            # --- Gráfico 4: Entrada U2 ---
        plt.subplot(2, 2, 4)
        plt.plot(self.num_hist, self.u2_hist, label='Input u2', color='g')
        plt.xlabel('Time Steps')
        plt.ylabel('Input u2')
        plt.title('Control Input u2')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    # Matriz de ganhos (com acoplamento)
    K_matrix = np.array([
        [1.0, 0.5],  # K11, K12
        [0.3, 1.2]   # K21, K22
    ])
    
    # Parâmetros
    wn1, zeta1 = 1.0, 0.7
    wn2, zeta2 = 0.8, 0.5
    
    # Setpoints
    sp = [5.0, 2.0]
    
    # Instancia o sistema
    env = TITOSystem(
        K=K_matrix,
        wn1=wn1, wn2=wn2,
        zeta1=zeta1, zeta2=zeta2,
        setpoint=sp,
        max_steps=500,
        render_mode='human'  # Ativa o render
    )
    
    env.reset()
    
    # Simula por 500 passos
    for i in range(5000):
        # Ação de controle (ex: degrau simples para teste)
        # Em um caso real, isso viria de um controlador (PID, RL, etc.)
        u1 = 1.0 if i > 20 else 0.0
        u2 = 0.5 if i > 50 else 0.0
        
        # Muda o setpoint no meio da simulação
        if i == 250:
            env.setpoint = [3.0, 4.0]

        obs, reward, terminated, truncated, info = env.step([u1, u2])
        
        if truncated:
            break
            
    # Renderiza os gráficos no final
    env.render()