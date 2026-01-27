
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
        self.integral += error*self.dt*self.Ki   
        derivative = (error - self.prev_error) / self.dt

        # PID
        u = self.Kp*error + self.integral + self.Kd*derivative

        # atualiza
        self.prev_error = error
        # saturação (se houver)
        if self.u_min is not None:
            u = max(self.u_min, u)
        if self.u_max is not None:
            u = min(self.u_max, u)

        return u