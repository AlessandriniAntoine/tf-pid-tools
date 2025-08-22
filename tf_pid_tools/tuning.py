"""
tuning.py
---------

Module providing PID tuning and optimization routines.

This module allows optimization of PID parameters for a given
system transfer function, using cost-based optimization and
heuristic methods (Ziegler–Nichols, IMC).
"""

import numpy as np
import control as ct
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from .pid import PID


def cost(params:list, sys_tf:ct.TransferFunction, pid: PID, weights: dict = {"time_response": 1, "transition_metric": 0, "steady_state_error": 0, "command_effort": 0.5}, t:np.ndarray=np.linspace(0, 20, 500)):
    """
    Compute the cost of a given PID configuration for a system.

    The cost is a weighted sum of performance metrics:
    - time response (settling time to 5% band),
    - transition behavior (overshoot + oscillations),
    - steady-state error,
    - command effort.

    Parameters
    ----------
    params : list
        Current PID parameters [Kp, Ki, Kd].
    sys_tf : control.TransferFunction
        System transfer function.
    pid : PID
        PID controller instance to update.
    weights : dict, optional
        Weights for the cost function components.
    t : np.ndarray, optional
        Time vector for simulation.

    Returns
    -------
    float
        Computed cost value.
    """
    pid.update_params(params, update_name=False)

    T = ct.feedback(pid.tf * sys_tf, 1)
    _, y = ct.step_response(T, t)

    # 1. time respone at 5%
    steady_state = y[-1]
    lower, upper = 0.95 * steady_state, 1.05 * steady_state
    try:
        last_out = np.max(np.where((y < lower) | (y > upper))[0])
        t95 = t[last_out + 1]
    except (ValueError, IndexError):
        t95 = 20

    # 2. Transition Behavior
    overshoot = np.max(y) - steady_state
    oscillations = np.sum(np.abs(y - steady_state)) * (t[1] - t[0])
    transition_metric = overshoot + 0.1 * oscillations

    # 3. Steady-state error
    steady_state_error = abs(1 - steady_state)

    # 4. Command effort
    e = 1 - y
    dt = t[1] - t[0]
    u = pid.compute_command_batch(e, dt)
    cmd_amp = np.max(np.abs(u))
    cmd_rate = np.max(np.abs(np.diff(u) / dt))
    command_metric = cmd_amp + 0.01 * cmd_rate


    return ( weights["time_response"] * t95 +
            weights["transition_metric"] * transition_metric +
            weights["steady_state_error"] * steady_state_error +
            weights["command_effort"] * command_metric )

def optimize(sys_tf:ct.TransferFunction, pid:PID, weights:dict, bounds:list | None=None, method:str='L-BFGS-B', t:np.ndarray=np.linspace(0, 20, 500)) -> tuple[PID, float]:
    """
    Optimize PID parameters to minimize the cost function.

    Parameters
    ----------
    sys_tf : control.TransferFunction
        Transfer function of the system.
    pid : PID
        Initial PID controller instance.
    weights : dict
        Weights for the cost function components.
    bounds : list, optional
        Bounds for the PID parameters (non-negative).
    method : str, optional
        Optimization method (default: 'L-BFGS-B').
    t : np.ndarray, optional
        Time vector for simulation.

    Returns
    -------
    tuple of (PID, float)
        Optimized PID controller and the achieved cost value.
    """

    required_keys = ["time_response", "transition_metric", "steady_state_error", "command_effort"]
    for key in required_keys:
        if key not in weights:
            weights[key] = 0.0
    if all(value == 0 for value in weights.values()):
        weights["time_response"] = 1.0

    initial_params = pid.get_params()

    if bounds is None:
        bounds = [(0, None)] * len(initial_params)
    else:
        for bound in bounds:
            if bound[0] < 0:
                raise ValueError("Bounds must be non-negative.")

    res = minimize(cost, initial_params, args=(sys_tf, pid, weights, t), bounds=bounds, method=method)
    pid.update_params(res.x)
    return pid, res.fun


def guess_pid(sys_tf: ct.TransferFunction) -> PID:
    """
    Estimate an initial PID controller using heuristics.

    The function first tries the Ziegler–Nichols method. If no
    sustained oscillations are detected, it falls back to IMC
    tuning. If both fail, it returns a default PID.

    Parameters
    ----------
    sys_tf : control.TransferFunction
        Transfer function of the system.

    Returns
    -------
    PID
        An initial guess for the PID controller.
    """
    Ku, Tu = None, None
    t = np.linspace(0, 100, 5000)

    # 1. Try Ziegler–Nichols
    Kp_test = 1e-3
    while Kp_test < 1e6:
        pid_test = PID(Kp_test, 0, 0)
        T_loop = ct.feedback(pid_test.tf * sys_tf, 1)
        _, y = ct.step_response(T_loop, t)

        peaks_idx = np.where((y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]))[0] + 1
        if len(peaks_idx) >= 3:
            Ku = Kp_test
            Tu = np.mean(np.diff(t[peaks_idx[:3]]))
            break

        Kp_test *= 1.5

    if Ku is not None and Tu is not None:
        # Formule PID classique ZN
        Kp = 0.6 * Ku
        Ki = 1.2 * Ku / Tu
        Kd = 0.075 * Ku * Tu
        return PID(Kp, Ki, Kd)

    # 2. Fallback IMC tuning

    num, den = ct.tfdata(sys_tf)
    num = np.array(num, dtype=float).flatten()
    den = np.array(den, dtype=float).flatten()

    if len(den) == 2:  # Premier ordre pur
        k = num[0] / den[-1]
        tau = den[0] / den[-1]
        lam = tau / 2
        Kp = tau / (k * lam)
        Ki = 1.0 / lam
        Kd = 0.0
        return PID(Kp, Ki, Kd)

    # 3. Full fallback
    return PID(1.0, 0.1, 0.05)


def auto_tune(sys_tf:ct.TransferFunction, initial_pid:PID, weights:dict={}, bounds:list|None=None, method:str='L-BFGS-B', plot:bool=True, verbose:bool=True) -> PID:
    """
    Automatically tune the PID controller for a given system.

    This function optimizes the PID parameters, prints the results,
    and optionally plots the closed-loop step response.

    Parameters
    ----------
    sys_tf : control.TransferFunction
        Transfer function of the system.
    initial_pid : PID
        Initial PID controller instance.
    weights : dict, optional
        Weights for the cost function components.
    bounds : list, optional
        Bounds for the PID parameters (non-negative).
    method : str, optional
        Optimization method (default: 'L-BFGS-B').
    plot : bool, optional
        Whether to plot the step response (default: True).
    verbose : bool, optional
        Whether to print optimization results (default: True).

    Returns
    -------
    PID
        Optimized PID controller.

    Examples
    --------
    >>> import control as ct
    >>> import numpy as np
    >>> from tf_pid_tools import PID, guess_pid, auto_tune
    >>>
    >>> # Define a simple second-order system
    >>> sys = ct.tf([1], [1, 2, 1])
    >>>
    >>> # Initial PID (can be a rough guess)
    >>> pid0 = guess_pid(sys)
    >>>
    >>> # Tune automatically
    >>> pid_opt = auto_tune(sys, pid0, weights={"time_response": 1, "command_effort": 0.5}, plot=False)
    """
    pid, cost_value = optimize(sys_tf, initial_pid, weights, bounds, method)
    T_PID = ct.feedback(pid.tf * sys_tf, 1)

    if verbose:
        print(f"Optimized PID parameters: Kp={pid.Kp:.2f}, Ki={pid.Ki:.2f}, Kd={pid.Kd:.2f}")
        print(f"Cost value: {cost_value:.2f}")
        print(f"Transfer function of the closed-loop system: \n{T_PID}")

    if plot:
        t = np.linspace(0, 20, 500)
        t_OL, y_OL = ct.step_response(sys_tf, t)
        t_PID, y_PID = ct.step_response(T_PID, t)
        plt.figure()
        plt.plot(t_OL, y_OL, '--r', label="Open Loop", alpha=0.9)
        plt.plot(t_PID, y_PID, '--', label=f"Closed Loop (Kp={pid.Kp:.2f}, Ki={pid.Ki:.2f}, Kd={pid.Kd:.2f})", alpha=0.6)
        plt.xlabel("Time (s)")
        plt.ylabel("Response")
        plt.title(f"Step Response - PID - cost={cost_value:.2f}")
        plt.grid(True)
        plt.legend()
        plt.show()

    return pid
