import control as ct
from tf_pid_tools import PID, auto_tune, guess_pid


wn = 1.0
zeta = 0.4
sys_tf = ct.tf([wn**2], [1, 2*zeta*wn, wn**2])


pid = guess_pid(sys_tf)
pid.print()

bounds = [(0, None) * len(pid.get_params())]
auto_tune(
    sys_tf=sys_tf,
    initial_pid=pid,
    weights={"time_response": 1, "transition_metric": 1, "steady_state_error": 5, "command_effort": 0.1},
    bounds=bounds,
    method='L-BFGS-B',
    plot=True,
    verbose=True
)
