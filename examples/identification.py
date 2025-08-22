import numpy as np
import matplotlib.pyplot as plt
import control as ct
from tf_pid_tools import estimate_transfer_function


wn = 1.0
zeta = 0.1
sys_tf = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])
print(f"System transfer function:\n{sys_tf}")

dt = 0.01
stop = 40
t = np.arange(0, stop + dt/2, dt)  # +dt/2 to ensure stop is included
u = 1.3 * np.ones_like(t)
_, signal = ct.forced_response(sys_tf, t, u)

# 3. Estimate
initial_tf = ct.TransferFunction([1], [1, 1, 1])
tf_est = estimate_transfer_function(u, signal, initial_tf, dt)
_, y = ct.forced_response(tf_est, t, u)
print(f"Identified transfer function:\n{tf_est}")
print(f"Error between estimated and actual signal: {signal - y}")

# 4. plot
plt.figure()
plt.plot(t, signal, '--r', label="Système (G1)", alpha=0.9)
plt.plot(t, y, '--b', label="Estimation", alpha=0.9)
plt.xlabel("Time (s)")
plt.ylabel("Response")
plt.title("Step Response")
plt.grid(True)
plt.legend()
plt.show()
