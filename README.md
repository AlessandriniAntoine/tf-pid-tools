# `tf-pid-tools`

**Transfer Function Identification & PID Auto-Tuning in Python**

`tf-pid-tools` is a Python library that combines **transfer function identification from data** with **PID controller tuning and optimization**.
It is designed for engineers, researchers, and hobbyists working in control systems who need fast prototyping, automatic tuning, and an interactive UI to visualize results.

---

## ✨ Features

- **PID Controller Class**
  - Clean, reusable PID implementation with `Kp`, `Ki`, `Kd`
  - Easy simulation with `python-control` library

- **Transfer Function Identification**
  - Identify continuous-time transfer functions from input–output data
  - Optimize numerator & denominator coefficients using `scipy.optimize`
  - MSE-based cost function
  - Bound constraints to ensure stability

- **PID Auto-Tuning**
  - Ziegler–Nichols closed-loop method (when possible)
  - IMC-based tuning fallback
  - Custom cost function optimization with adjustable weight sliders
  - Interactive **Tkinter app** for tuning and visualization

---

## 📦 Installation

### From source (local development)
```bash
# Clone the repository
git clone https://github.com/yourusername/tf-pid-tools.git
cd tf-pid-tools

# Install in editable mode
pip install -e .
# Install in non-editable mode
pip install .
```

### PyPI installation
(Coming soon) — PyPI installation:
```bash
pip install tf-pid-tools
```

---

##  📊 Use Cases

- Automatic PID tuning for industrial systems
- Educational demos for control theory
- Rapid prototyping of controllers
- System identification from experimental data
- Comparing tuning strategies

---

## 🚀 Examples

See the [examples](examples/) directory for more details.
- PID tuning from a known transfer function
- Interactive PID tuning app
- Transfer function identification from synthetic or measured data
