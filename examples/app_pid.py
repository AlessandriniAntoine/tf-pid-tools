import control as ct
from tf_pid_tools import auto_tune_app


wn = 1.0
zeta = 0.4
sys_tf = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])

# Launch interactive GUI to tune PID
pid = auto_tune_app(sys_tf)
print(f"{pid.Kp=}, {pid.Ki=}, {pid.Kd=}")
