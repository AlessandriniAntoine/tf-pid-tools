import control as ct
from tf_pid_tools import PID, auto_tune, guess_pid


order = 2

# -----------------------------
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

# -----------------------------
# 2. Définition du PID
pid = guess_pid(sys_tf)
pid.print()

bounds = [(0, None), (0, None), (0, None)]
auto_tune(
    sys_tf=sys_tf,
    initial_pid=pid,
    weights={"time_response": 1, "transition_metric": 1, "steady_state_error": 10, "command_effort": 0.},
    bounds=bounds,
    method='L-BFGS-B',
    plot=True,
    verbose=True
)
