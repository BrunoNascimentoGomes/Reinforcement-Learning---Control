
# u = Kp * error + integral(error) + derivada(error)

class PID:
    def __init__(self, Kp, Ki, Kd, dt, u_min = 0, u_max = float('inf')):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt

        self.integral = 0
        self.prev_error = 0.0

        self.u_min = u_min
        self.u_max = u_max
    
    def compute(self, error):
        self.integral += error*self.dt   
        derivative = (error - self.prev_error) / self.dt

        # PID
        u = self.Kp*error + self.Ki*self.integral + self.Kd*derivative

        # atualiza
        self.prev_error = error
        # saturação (se houver)
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)

        return u

class PID_Velo_Form:
    def __init__(self, Kc, tau_i, tau_d=0.0, dt=4.0, u_min=-float('inf'), u_max=float('inf'), u_init=0.0):
        self.Kc = Kc  # Ganho proporcional
        self.tau_i = tau_i  # Tempo de integral
        self.tau_d = tau_d  # Tempo derivativo
        self.dt = dt  # Intervalo de amostragem

        self.u = u_init  # Valor inicial do controle
        self.u_min = u_min  # Valor mínimo de controle
        self.u_max = u_max  # Valor máximo de controle

        # Erros passados
        self.e_k1 = 0.0  # Erro anterior (k-1)
        self.e_k2 = 0.0  # Erro dois passos atrás (k-2)

    def compute_control(self, e):
        Ts = self.dt

        # Fórmula do PID em forma de velocidade (adaptada da equação de Seborg et al. 2016)
        delta_u = self.Kc * (
            (1 + (Ts / self.tau_i) + (self.tau_d / Ts)) * e - 
            ((self.tau_d / Ts) + 1) * self.e_k1 + 
            (self.tau_d / Ts) * self.e_k2
        )

        # Atualiza o controle (u)
        self.u += delta_u

        # Aplicar saturação (limitar u entre u_min e u_max)
        self.u = max(self.u_min, min(self.u_max, self.u))

        # Atualiza os erros
        self.e_k2 = self.e_k1
        self.e_k1 = e

        return self.u

