import numpy as np
from scipy.optimize import minimize
import control as ct


def cost(params:list, inputs:np.ndarray, outputs:np.ndarray, nb_numerator:int, dt:float):
    num = params[:nb_numerator]
    den = params[nb_numerator:]
    tf = ct.TransferFunction(num, den)
    stop = dt * (outputs.shape[0]-1)
    t, y = ct.forced_response(tf, np.arange(0, stop + dt/2, dt), inputs)
    return np.mean((outputs - y) ** 2)  # MSE


def estimate_transfer_function(inputs:np.ndarray, outputs:np.ndarray, initial_tf:ct.TransferFunction, dt:float) -> ct.TransferFunction:
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
