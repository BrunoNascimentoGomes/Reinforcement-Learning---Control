import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class ModeloSegundaOrdem:
    def __init__(self, K, wn, zeta, dt=0.01, setpoint=0, max_steps=1000, render_mode = None, tol = 1e-3):
        
        #Parameters
        self.K = K
        self.wn = wn
        self.zeta = zeta
        self.dt = dt
        self.setpoint = setpoint

        #Episode Config
        self.max_steps = int(max_steps)
        self.render_mode = render_mode 

        #Termination tolerance
        self.tol = float(tol)

        self.state = np.array([0.0, 0.0])  # x1 = y, x2 = dydt

        self.t = 0.0
        self.num_step = 0

        # For plot/render
        self.y_hist = []
        self.u_hist = []
        self.num_hist = []
        self.setpoint_hist = []
        self.error_hist = []
    
    def reset(self, x1 = 0.0, x2 = 0.0):

        self.state = np.array([0.0, 0.0], dtype=float)
        self.t = 0.0
        self.num_step = 0
        self.y_hist = []
        self.u_hist = []
        self.num_hist = []
        self.setpoint_hist = []
        self.error_hist = []

        

        obs = np.array([self.setpoint - 0.0, 0.0, 0.0, self.setpoint], dtype=np.float32)
        return obs, {}

    
    #ODE
    def _ode(self,t , x, u):
        x1, x2 = x
        dx1dt = x2
        dx2dt = -2*self.zeta*self.wn*x2 - (self.wn**2)*x1 + (self.wn**2)*self.K*u
        return [dx1dt, dx2dt]
    
    def step(self, u):
        u = np.asarray(u, dtype=float).reshape(-1)[0]
        sol = solve_ivp(self._ode, [self.t, self.t + self.dt], self.state, t_eval=[self.t + self.dt], args=(u,), method='RK45')
        
        self.state = sol.y[:, -1]
        self.t += self.dt
        self.num_step += 1

        y = float(self.state[0])
        error = float(self.setpoint - y)

        #Delta error e delta u
        delta_e = error - (self.error_hist[-1] if self.error_hist else 0.0)
        delta_u = u - (self.u_hist[-1] if self.u_hist else 0.0)

        #Delta_y
        delta_y = y - self.y_hist[-1] if self.y_hist else 0.0
        delta_y_2 = delta_y ** 2

        self.num_hist.append(self.num_step)
        self.y_hist.append(y)
        self.u_hist.append(u)
        self.setpoint_hist.append(self.setpoint)
        self.error_hist.append(error)
        
        # observation & reward
        obs = np.array([error, y, delta_e, self.setpoint], dtype=np.float32)
        
        fator_penalidade = 0.01
        #Mudei a recompensa para penalizar não admitir mudanças bruscas de
        #u. Original: reward = -float(error ** 2) if abs(error) > self.tol else 10
        reward = -abs(float(error)) - fator_penalidade*float(delta_u**2) if abs(error) > self.tol else 10 
        
        #reward = -float(error ** 2) - fator_penalidade*float(delta_u**2) if abs(error) > self.tol else 10

        
        #Terminated or Truncated
        terminated = False
        truncated = self.num_step >= self.max_steps

        info = {
            "time": float(self.t),
            "u_applied": u,
            "y": y,
        }


        return obs, reward, terminated, truncated, info
    
    def render(self, mode="human"):
        # Simple plot of y vs step
        if len(self.num_hist) == 0:
            print("Nothing to render (no steps yet).")
            return

        plt.figure(figsize=(8, 4))
        plt.plot(self.num_hist, self.y_hist, label="y (plant output)")
        plt.plot(self.num_hist, self.setpoint_hist, '--', color='r', label="Setpoint")  # <=== Adicionado no gráfico
        plt.xlabel("Step")
        plt.ylabel("y / Setpoint")
        plt.title("Plant Response")
        plt.grid(True)
        plt.legend()
        
        # Plot u(t)
        plt.figure(figsize=(8, 4))
        plt.plot(self.num_hist, self.u_hist, color="g", label="u (control action)")
        plt.xlabel("Step")
        plt.ylabel("u")
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()
    

if __name__ == "__main__":
    env = ModeloSegundaOrdem(K=1.0, wn=2.0, zeta=0.3, dt=0.02, setpoint=0.0, max_steps=1000)

# Reinicia o ambiente
    obs, _ = env.reset()

# Sinal de controle constante para simplificar
    u = 1.0

# Loop de simulação
    for step in range(1000):
        u = 1
    # Muda o setpoint durante a simulação
        if step == 300:
            env.setpoint = 1.0
        elif step == 600:
            env.setpoint = 2.0

    # Executa um passo no modelo
        obs, reward, terminated, truncated, info = env.step(u)

# Mostra o resultado (deve exibir a curva y e o setpoint variando)
    env.render()