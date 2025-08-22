"""
identification.py
-----------------

Module providing system identification tools.

This module allows estimation of a transfer function model
from input/output data using optimization (mean squared error minimization).
"""

import numpy as np
from scipy.optimize import minimize
import control as ct


def cost(params:list, inputs:np.ndarray, outputs:np.ndarray, nb_numerator:int, dt:float):
    """
    Compute the mean squared error between the measured output
    and the response of a candidate transfer function.

    Parameters
    ----------
    params : list
        Flattened parameters of the transfer function
        (numerator coefficients followed by denominator coefficients).
    inputs : np.ndarray
        Input signal applied to the system.
    outputs : np.ndarray
        Measured output signal of the system.
    nb_numerator : int
        Number of numerator coefficients.
    dt : float
        Sampling time step.

    Returns
    -------
    float
        Mean squared error (MSE) between measured and simulated outputs.
    """
    num = params[:nb_numerator]
    den = params[nb_numerator:]
    tf = ct.TransferFunction(num, den)
    stop = dt * (outputs.shape[0]-1)
    t, y = ct.forced_response(tf, np.arange(0, stop + dt/2, dt), inputs)
    return np.mean((outputs - y) ** 2)  # MSE


def estimate_transfer_function(inputs:np.ndarray, outputs:np.ndarray, initial_tf:ct.TransferFunction, dt:float) -> ct.TransferFunction:
    """
    Estimate a transfer function from input/output data.

    The estimation is done by minimizing the mean squared error (MSE)
    between the measured output and the simulated output of a transfer function.

    Parameters
    ----------
    inputs : np.ndarray
        Input signal applied to the system.
    outputs : np.ndarray
        Measured output signal of the system.
    initial_tf : control.TransferFunction
        Initial guess for the transfer function.
    dt : float
        Sampling time step.

    Returns
    -------
    control.TransferFunction
        Estimated transfer function.

    Examples
    --------
    >>> import numpy as np
    >>> import control as ct
    >>> from tf_pid_tools import estimate_transfer_function
    >>>
    >>> # True system: first-order transfer function G(s) = 1 / (s+1)
    >>> sys_true = ct.tf([1], [1, 1])
    >>> t = np.linspace(0, 10, 100)
    >>> u = np.ones_like(t)  # Step input
    >>> _, y_true, _ = ct.forced_response(sys_true, t, u)
    >>>
    >>> # Add a bit of noise to simulate measurements
    >>> y_meas = y_true + 0.05 * np.random.randn(len(y_true))
    >>>
    >>> # Initial guess: G(s) = 0.5 / (s+2)
    >>> sys_init = ct.tf([0.5], [1, 2])
    >>>
    >>> # Estimate transfer function from data
    >>> sys_est = estimate_transfer_function(u, y_meas, sys_init, dt=t[1]-t[0])
    >>> print(sys_est)
    """
    initial_numerator = initial_tf.num[0][0].tolist()
    initial_denominator = initial_tf.den[0][0].tolist()
    initial_params = initial_numerator + initial_denominator
    bounds = [(None, None)] * len(initial_numerator) + [(1e-4, None)] * len(initial_denominator)
    res = minimize(cost, initial_params,
        args=(inputs, outputs, len(initial_numerator), dt),
        method='L-BFGS-B', bounds=bounds)
    num = res.x[:len(initial_numerator)]
    den = res.x[len(initial_numerator):]
    return ct.TransferFunction(num, den)
