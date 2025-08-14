import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import control as ct
from .tuning import optimize, guess_pid
from .identification import estimate_transfer_function


def auto_tune_app(sys_tf: ct.TransferFunction, bounds: list | None=None, method: str='L-BFGS-B', dt: float=0.01):
    initial_pid = guess_pid(sys_tf)
    current_pid = initial_pid.copy()

    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title("PID Auto-tuning")

    end_time = tk.DoubleVar(value=10.0)
    t = np.arange(0, end_time.get(), dt)
    _, y_sys = ct.step_response(sys_tf, t)

    goal = y_sys[-1]

    # Figure Matplotlib
    fig, ax = plt.subplots(figsize=(6,4))
    T_init = ct.feedback(initial_pid.tf * sys_tf, 1)
    _, y_init = ct.step_response(goal * T_init, t)
    line_sys, = ax.plot(t, y_sys, '--r', label="Système")
    line_pid, = ax.plot(t, y_init, label="PID optimisé")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Réponse")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"PID initial: \nKp={initial_pid.Kp}, Ki={initial_pid.Ki}, Kd={initial_pid.Kd}")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, pady=10)

    # Sliders et labels
    weights = {
        "time_response": tk.DoubleVar(value=1.0),
        "transition_metric": tk.DoubleVar(value=1.0),
        "steady_state_error": tk.DoubleVar(value=1.0),
        "command_effort": tk.DoubleVar(value=0.0)
    }

    def optimize_callback():
        nonlocal current_pid

        t = np.arange(0, end_time.get(), dt)
        _, y_sys = ct.step_response(sys_tf, t)

        w = {k.lower().replace(" ", "_"): v.get() for k,v in weights.items()}
        pid, cost = optimize(sys_tf, initial_pid.copy(), w, bounds, method, t)
        current_pid = pid.copy()
        T_PID = ct.feedback(pid.tf * sys_tf, 1)
        _, y_PID = ct.step_response(goal * T_PID, t)
        line_sys.set_xdata(t)
        line_sys.set_ydata(y_sys)
        line_pid.set_xdata(t)
        line_pid.set_ydata(y_PID)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(
            f"{pid.name} optimized: Cost: {cost:.1e}\nKp={pid.Kp:.1e}, Ki={pid.Ki:.1e}, Kd={pid.Kd:.1e}"
        )
        canvas.draw_idle()

    def make_slider(name, var, row, max):
        label = ttk.Label(root, text=name)
        label.grid(row=row, column=0, sticky="w", padx=5)

        slider = ttk.Scale(root, from_=0, to=max, variable=var, orient="horizontal")
        slider.grid(row=row, column=1, sticky="ew", padx=5)

        value_label = ttk.Label(root, text=f"{var.get():.2f}")
        value_label.grid(row=row, column=2, sticky="e", padx=5)

        def on_slide(event):
            value_label.config(text=f"{var.get():.2f}")

        def on_release(event):
            value_label.config(text=f"{var.get():.2f}")
            optimize_callback()

        slider.bind("<Motion>", on_slide)
        slider.bind("<ButtonRelease-1>", on_release)

    def on_close():
        root.quit()
        root.destroy()
        plt.close(fig)

    make_slider("Time response", weights["time_response"], 1, 10)
    make_slider("Transition", weights["transition_metric"], 2, 10)
    make_slider("Steady state error", weights["steady_state_error"], 3, 10)
    make_slider("Command effort", weights["command_effort"], 4, 10)
    make_slider("Duration simulation (s)", end_time, 5, 500)

    root.columnconfigure(1, weight=1)
    optimize_callback()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    return current_pid


def auto_estimate_app(inputs, outputs, dt):
    current_tf = ct.TransferFunction([1], [1])

    root = tk.Tk()
    root.title("PID Auto-tuning")

    t = np.arange(0, (len(outputs)-1)*dt + dt/ 2 , dt)


    # Figure Matplotlib
    fig, ax = plt.subplots(figsize=(10,8))
    _, y_init = ct.forced_response(current_tf, t, inputs)
    line_sys, = ax.plot(t, outputs, '--r', label="Système")
    line_pid, = ax.plot(t, y_init, label="PID optimisé")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Réponse")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"TF : Error = 0\nNum={current_tf.num[0][0].tolist()}, \nDen={current_tf.den[0][0].tolist()}")

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=3, pady=10)

    # Sliders et labels
    weights = {
        "number_of_zeros": tk.DoubleVar(value=0.0),
        "number_of_poles": tk.DoubleVar(value=1.0),
    }


    def optimize_callback():
        nonlocal current_tf
        initial_tf = ct.TransferFunction([1] * int(weights["number_of_zeros"].get()+1), [1] * int(weights["number_of_poles"].get()+1))
        tf = estimate_transfer_function(inputs, outputs, initial_tf, dt)
        num = tf.num[0][0].tolist()
        den = tf.den[0][0].tolist()
        current_tf = tf.copy()
        _, y_tf = ct.forced_response(tf, t, inputs)
        error = np.linalg.norm(outputs - y_tf)
        line_pid.set_ydata(y_tf)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(
            f"TF : Error = {error:.2e}\nNum=[{', '.join(f'{n:.1e}' for n in num)}],\nDen=[{', '.join(f'{d:.1e}' for d in den)}]"
        )
        canvas.draw_idle()

    def make_slider(name, var, row):
        label = ttk.Label(root, text=name)
        label.grid(row=row, column=0, sticky="w", padx=5)

        slider = ttk.Scale(root, from_=0, to=5, variable=var, orient="horizontal")
        slider.grid(row=row, column=1, sticky="ew", padx=5)

        value_label = ttk.Label(root, text=f"{var.get():.2f}")
        value_label.grid(row=row, column=2, sticky="e", padx=5)

        def on_slide(event):
            value_label.config(text=f"{var.get():.2f}")
            val = int(var.get())
            var.set(val)

        def on_release(event):
            val = int(var.get())
            var.set(val)
            value_label.config(text=str(val))
            optimize_callback()

        slider.bind("<Motion>", on_slide)
        slider.bind("<ButtonRelease-1>", on_release)

    def on_close():
        root.quit()
        root.destroy()
        plt.close(fig)

    make_slider("Number of zeros", weights["number_of_zeros"], 1)
    make_slider("Number of poles", weights["number_of_poles"], 2)


    root.columnconfigure(1, weight=1)
    optimize_callback()
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

    return current_tf
