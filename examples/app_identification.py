import numpy as np
import control as ct
from tf_pid_tools import auto_estimate_app


wn = 1.0
zeta = 0.1
sys_tf = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])

stop = 50
dt = 0.01
t = np.arange(0, stop+dt/2, dt)
# inputs = np.sin(2 * np.pi * 0.1 * t)
inputs = np.ones_like(t)
_, outputs = ct.forced_response(sys_tf, t, inputs)

# Launch interactive GUI to identify system
tf_est = auto_estimate_app(inputs, outputs, dt)
print(f"Identified transfer function:\n{tf_est}")
