import control as ct
from tf_pid_tools import auto_tune_app

order = 2

# 1. Définition de la FT du premier ordre
tau = 2
k = 1
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

auto_tune_app(sys_tf)
