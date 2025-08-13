import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import control as ct
from .tuning import optimize, guess_pid


def auto_tune_app(sys_tf: ct.TransferFunction):
    initial_pid = guess_pid(sys_tf)

    # Création de la fenêtre Tkinter
    root = tk.Tk()
    root.title("PID Auto-tuning")

    # Figure Matplotlib
    fig, ax = plt.subplots(figsize=(6,4))
    t = np.linspace(0, 10, 500)
    T_init = ct.feedback(initial_pid.tf * sys_tf, 1)
    _, y_init = ct.step_response(T_init, t)
    line_pid, = ax.plot(t, y_init, label="PID optimisé")
    line_sys, = ax.plot(*ct.step_response(sys_tf, t), '--r', label="Système")
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Réponse")
    ax.grid(True)
    ax.legend()
    ax.set_title(f"PID initial : Kp={initial_pid.Kp}, Ki={initial_pid.Ki}, Kd={initial_pid.Kd}")

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
        w = {k.lower().replace(" ", "_"): v.get() for k,v in weights.items()}
        pid, cost = optimize(sys_tf, initial_pid.copy(), w)
        T_PID = ct.feedback(pid.tf * sys_tf, 1)
        _, y_PID = ct.step_response(T_PID, t)
        line_pid.set_ydata(y_PID)
        ax.relim()
        ax.autoscale_view()
        ax.set_title(
            f"{pid.name} optimisé : Kp={pid.Kp:.2f}, Ki={pid.Ki:.2f}, Kd={pid.Kd:.2f} | Coût = {cost:.2f}"
        )
        canvas.draw_idle()

    def make_slider(name, var, row):
        label = ttk.Label(root, text=name)
        label.grid(row=row, column=0, sticky="w", padx=5)

        slider = ttk.Scale(root, from_=0, to=10, variable=var, orient="horizontal")
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

    make_slider("Time response", weights["time_response"], 1)
    make_slider("Transition", weights["transition_metric"], 2)
    make_slider("Steady state error", weights["steady_state_error"], 3)
    make_slider("Command effort", weights["command_effort"], 4)

    root.columnconfigure(1, weight=1)
    optimize_callback()
    root.mainloop()
