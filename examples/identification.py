import numpy as np
import matplotlib.pyplot as plt
import control as ct
from tf_pid_tools import estimate_transfer_function


# 1. Définition de la FT
tau = 2
k = 1
sys_tf1 = ct.tf([k], [tau, 1])  # k / (tau s + 1)

wn = 1.0   # pulsation naturelle
zeta = 0.1 # faible amortissement -> propice aux oscillations
sys_tf2 = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])

order = 2
if order == 1:
    sys_tf = sys_tf1
elif order == 2:
    sys_tf = sys_tf2
else:
    raise ValueError("Must be order one or two")
print(sys_tf)

# 2. Génération des données
dt = 0.01
stop = 40
t = np.arange(0, stop + dt/2, dt)  # +dt/2 pour être sûr d'inclure stop
u = 1.3 * np.ones_like(t)
_, signal = ct.forced_response(sys_tf, t, u)

# 3. Estimate
initial_tf = ct.TransferFunction([1], [1, 1, 1])
tf_est = estimate_transfer_function(u, signal, initial_tf, dt)
_, y = ct.forced_response(tf_est, t, u)
print(tf_est)

# 4. plot
plt.figure()
plt.plot(t, signal, '--r', label="Système (G1)", alpha=0.9)
plt.plot(t, y, '--b', label="Estimation", alpha=0.9)
plt.xlabel("Temps (s)")
plt.ylabel("Réponse")
plt.title("Réponse indicielle")
plt.grid(True)
plt.legend()
plt.show()
