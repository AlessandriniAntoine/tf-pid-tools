import numpy as np
import control as ct
from tf_pid_tools import auto_estimate_app

order = 2

# 1. Définition de la FT du premier ordre
tau = 2
k = 5
sys_tf1 = ct.tf([k], [tau, 1])  # k / (tau s + 1)

wn = 1.0   # pulsation naturelle
zeta = 0.1 # faible amortissement -> propice aux oscillations
sys_tf2 = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])

if order == 1:
    sys_tf = sys_tf1
elif order == 2:
    sys_tf = sys_tf2
else:
    raise ValueError("Must be order one or two")

stop = 50
dt = 0.01
t = np.arange(0, stop+dt/2, dt)
# inputs = np.sin(2 * np.pi * 0.1 * t)
inputs = np.ones_like(t)
_, outputs = ct.forced_response(sys_tf, t, inputs)

tf_est = auto_estimate_app(inputs, outputs, dt)
print(f"Identified transfer function:\n{tf_est}")
import matplotlib.pyplot as plt
plt.show()
