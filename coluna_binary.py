import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class DistilationColumn:
    def __init__(self, NT =41, NF = 21, alpha = 1.5, F = 1, zF = 0.5, qF = 1, M1 = 5, M2 = 0.5,
                M3 = 5, dt = 3, setpoint = [0.99,0.01], u =[2.706, 3.706], state = None,
                max_steps=1000, render_mode = "human", tol = 0):
        
        # Parameters
        self.NT = NT  # Number of trays
        self.NF = NF  # Feed tray
        self.F = F    # Feed flow rate
        self.zF = zF  # Feed composition
        self.qF = qF  # Feed thermal condition
        self.M1 = M1  # Holdup in the receiver
        self.M2 = M2  # Holdup in the feed tray
        self.M3 = M3  # Holdup in the reboiler
        self.alpha = alpha  # Relative volatility
        
        
        self.setpoint = setpoint # [xD_setpoint, xB_setpoint] - xD topo, xB fundo
        self.u = u   # [LR, VS] - Reflux and boilup rates
        self.calculo_flow() # Calculate flow rates based on u

        #Indexes 
        self.i_cond = 0
        self.i_feed = NF - 1
        self.i_reb = NT - 1

        #Initial State
        if state is None:
            self.x = np.zeros(NT)   #Guarda as concentraes em cada bandeja
        else:
            self.x = state      #Guarda as concentraes iniciais (x)
            
        

        # Simulation parameters
        self.dt = dt
        self.max_steps = int(max_steps)
        self.render_mode = render_mode
        self.num_step = 0
        self.t = 0.0
        self.tol = float(tol)

        # For plot/render
        self.xd_hist = []
        self.xb_hist = []
        
        self.ud_hist = []
        self.ub_hist = []
        
        self.num_hist = []
        self.setpoint_xd_hist = []
        self.setpoint_xb_hist = []

        self.errord_hist = []
        self.errorb_hist = []
        
    def equilibrio(self, x):
        # Simplex model for equilibrium
        y = self.alpha * x / (1 + (x * (self.alpha - 1)))
        return y

    def calculo_flow(self):
        self.Lr = self.u[0]  # Reflux rate - Topo
        self.Vs = self.u[1]  # Boilup rate - Fundo

        self.Ls = self.Lr + self.F * self.qF  # Liquid flow rate in the stripping section
        self.Vr = self.Vs + self.F * (1 - self.qF)  # Vapor flow rate in the rectifying section

        self.B = self.Ls - self.Vs  # Bottoms flow rate
        self.D = self.Vr - self.Lr  # Distillate flow rate
    
    def _ode(self, t, x):
        dxdt = np.zeros(self.NT)

        y = self.equilibrio(x)
        # Receiver
        dxdt[self.i_cond] = (self.Vr * y[1] - self.Vr * x[0]) / self.M1

        # Rectifying section
        for i in range(self.i_cond + 1, self.i_feed):
            dxdt[i] = (self.Lr * x[i-1] + self.Vr * y[i+1] - self.Lr * x[i] - self.Vr * y[i]) / self.M2

        # Feed tray
        dxdt[self.i_feed] = (self.Lr*x[self.i_feed-1]+self.Vs*y[self.i_feed+1]+self.F*self.zF-self.Ls*x[self.i_feed]-self.Vr*y[self.i_feed])/self.M2

        # Stripping section
        for i in range(self.i_feed + 1, self.i_reb):
            dxdt[i] = (self.Ls*x[i-1] + self.Vs*y[i+1] - self.Ls*x[i] -self.Vs*y[i])/self.M2

        # Reboiler
        dxdt[self.i_reb] = (self.Ls*x[self.i_reb-1] -self.B * x[self.i_reb] - self.Vs*y[self.i_reb]) / self.M3

        return dxdt

    def step(self, u):
        self.u = u
        self.calculo_flow()

        sol = solve_ivp(self._ode, [self.t, self.t+self.dt], self.x,t_eval=[self.t + self.dt], method='RK45')
        self.x = sol.y[:, -1]
        

        xD = self.x[0]
        xB = self.x[-1]
        '''
        Setpoint[0] - xD (topo)
        Setpoint[1] - xB (fundo)
        '''        
        error_D = (self.setpoint[0] - xD)
        error_B = (self.setpoint[1] - xB)

        self.t += self.dt
        self.num_step += 1

        self.xd_hist.append(xD)
        self.xb_hist.append(xB)

        '''
        Lr = self.u[0]  # Reflux rate - Topo
        Vs = self.u[1]  # Boilup rate - Fundo
        '''
        self.ud_hist.append(self.u[0])
        self.ub_hist.append(self.u[1])

        self.num_hist.append(self.num_step)
        self.setpoint_xd_hist.append(self.setpoint[0])
        self.setpoint_xb_hist.append(self.setpoint[1])
        self.errord_hist.append(error_D)
        self.errorb_hist.append(error_B)

        reward_d = - (abs(error_D)*10) if abs(error_D) > self.tol else 10
        reward_b = - (abs(error_B)*10) if abs(error_B) > self.tol else 10

        reward_somada = reward_d + reward_b

        obsd = np.array([error_D, xD, self.ub_hist[-1], self.setpoint[0]], dtype=np.float32)
        obsb = np.array([error_B, xB, self.ud_hist[-1], self.setpoint[1]], dtype=np.float32)
        obs_concatenated = np.array([error_D, xD, self.ub_hist[-1], self.setpoint[0],
                                     error_B, xB, self.ud_hist[-1], self.setpoint[1]], dtype=np.float32)
        '''LEMBRAR QUE OBS E REWARD SAO TUPLAS PARA CADA CONTROLE'''
        obs = (obsd, obsb, obs_concatenated)   
        
        reward = (reward_d, reward_b, reward_somada)
        

        terminated = False
        truncated = self.num_step >= self.max_steps
        info = {}
        return obs, reward, terminated, truncated, info
    
    def render(self, render_mode=None):
        if render_mode is None:
            render_mode = self.render_mode

        if render_mode == "human":
            plt.figure(figsize=(12, 6))

            plt.subplot(2, 1, 1)
            plt.plot(self.num_hist, self.xd_hist, label='xD (Top Composition)')
            plt.plot(self.num_hist, self.setpoint_xd_hist, 'r--', label='xD Setpoint')
            plt.xlabel('Time Step')
            plt.ylabel('Composition - XD')
            plt.title('Distillation Column Compositions (XD)')
            plt.legend()
            plt.grid()
            
            
            plt.subplot(2, 1, 2)
            plt.plot(self.num_hist, self.xb_hist, label='xB (Bottom Composition)')
            plt.plot(self.num_hist, self.setpoint_xb_hist, 'r--', label='xB Setpoint')
            plt.xlabel('Time Step')
            plt.ylabel('Composition - XB')
            plt.title('Distillation Column Compositions (XB)')
            plt.legend()
            plt.grid()

            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            plt.plot(self.num_hist, self.ud_hist, label='Reflux Rate (LR)')
            plt.xlabel('Time Step')
            plt.ylabel('Flow Rates')
            plt.title('Control Actions - LR - Controla o do Topo(XD)')
            plt.legend()
            plt.grid()

            plt.subplot(2, 1, 2)
            plt.plot(self.num_hist, self.ub_hist, label='Boilup Rate (VS)')
            
            plt.xlabel('Time Step')
            plt.ylabel('Flow Rates')
            plt.title('Control Actions - VS - Controla o Fundo(XB)')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()
    
    def plot_estagio(self):
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.NT + 1), self.x, marker='o')
        plt.xlabel('Tray Number')
        plt.ylabel('Liquid Composition (x)')
        plt.title('Composition Profile Along the Distillation Column')
        plt.grid()
        plt.show()
    
    def plot_u(self):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(self.num_hist, self.ud_hist, label='Reflux Rate (LR)')
        plt.xlabel('Time Step')
        plt.ylabel('Flow Rates')
        plt.title('Control Actions - LR - Controla o do Topo(XD)')
        plt.legend()
        plt.grid()

        plt.subplot(2, 1, 2)
        plt.plot(self.num_hist, self.ub_hist, label='Boilup Rate (VS)')
            
        plt.xlabel('Time Step')
        plt.ylabel('Flow Rates')
        plt.title('Control Actions - VS - Controla o Fundo(XB)')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

    def reset(self,state = None):
        '''
        self.x =np.array([0.989996,   0.98506869, 0.97890509, 0.97123076, 0.96173051, 0.95005371,
 0.93582724, 0.91867871, 0.89827236, 0.87435801, 0.84683005, 0.81578814,
 0.78158594, 0.74485109, 0.70646163, 0.66747398, 0.6290119,  0.59213988,
 0.55775034, 0.52648847, 0.49872563, 0.47417117, 0.4455474,  0.41300484,
 0.37704334, 0.33853025, 0.29864485, 0.25874749, 0.22020013, 0.18418648,
 0.15157992, 0.12288666, 0.09826274, 0.07758201, 0.0605254,  0.04666715,
 0.03554402, 0.02670329, 0.01973121, 0.01426654, 0.010004])
    
 '''
        if state is None:
            self.x = np.zeros(self.NT)   #Guarda as concentraes em cada bandeja
        else:
            self.x = state      #Guarda as concentraes iniciais (x)
        
        self.t = 0.0
        self.num_step = 0

        self.xd_hist = []
        self.xb_hist = []
        
        self.ud_hist = []
        self.ub_hist = []
        
        self.num_hist = []
        self.setpoint_xd_hist = []
        self.setpoint_xb_hist = []

        error_D = self.setpoint[0] - self.x[0]
        error_B = self.setpoint[1] - self.x[-1]


        u =[2.706, 3.706]
        self.u = u
        #obs = np.array([error_D, self.x[0], error_B, self.x[-1]], dtype=np.float32)
        obsd = np.array([error_D, self.x[0], u[1],self.setpoint[0]], dtype=np.float32)
        obsb = np.array([error_B, self.x[-1], u[0],self.setpoint[1]], dtype=np.float32)
        _ = 0
        obs = (obsd, obsb, _)
        return obs, {}
if __name__ == "__main__":
    Lr = 2.706
    #Lr += 0.01 
    Vs = 3.206
    env = DistilationColumn(render_mode="human")
    obs = env.reset()
    #print(env.x)
    
    for _ in range(100000):
        obs = env.step([Lr, Vs])

    #print(env.xd_hist[-1], env.xb_hist[-1])
    print(env.x)
    env.plot_estagio()
    env.render()