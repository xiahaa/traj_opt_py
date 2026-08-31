# traj_opt_py

Endpoint-based minimum-jerk trajectory optimization utilities in Python.

## Features

- Solve the endpoint trajectory QP with `qpsolvers`
- Build equality and inequality constraints for position, velocity, and acceleration
- Extract polynomial coefficients and sample the resulting path
- Includes a self-contained demo script

## Install

```bash
pip install -e .
pip install -e .[demo]
```

## Demo

Run the example script:

```bash
python examples/demo.py
```

It writes `examples/demo_result.npz` and, when `matplotlib` is installed, `examples/demo_result.png`.

## Usage

```python
import numpy as np
import traj_opt

x = traj_opt.example1(
    p=np.zeros((4, 3)),
    t=np.arange(4, dtype=float),
    p_cons=np.array([[0.5, 0.1, 0.0], [1.5, 0.2, 0.1]]),
    t_cons=np.array([0.5, 1.5]),
)
```

## Repository layout

- `traj_opt.py` – core optimizer helpers
- `examples/demo.py` – runnable demo
- `testtraj.ipynb` – exploratory notebook

## Dependencies

- numpy
- scipy
- qpsolvers
- osqp
- matplotlib (demo only)
