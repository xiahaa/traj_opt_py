"""Self-contained demo for the trajectory optimizer."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import traj_opt


def _reference_path(times: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            times,
            0.35 * np.sin(np.pi * times / times[-1]),
            0.2 * np.cos(np.pi * times / times[-1]),
        ]
    )


def main() -> None:
    times = np.linspace(0.0, 3.0, 7)
    constraints = _reference_path(times)

    x = traj_opt.example1(
        p=np.zeros((4, 3), dtype=float),
        t=np.arange(4, dtype=float),
        p_cons=constraints[1:-1],
        t_cons=times[1:-1],
        l=0.02,
        mu=1.0,
        tol=0.05,
    )

    if x is None:
        raise RuntimeError("trajectory optimization failed")

    poly = traj_opt.get_polynomial_coefficients(x)
    xyz, _ = traj_opt.get_traj_pts(poly, num_pts_per_seg=150)

    output = Path(__file__).with_name("demo_result.npz")
    np.savez(output, optimized_path=xyz, constraints=constraints)
    print(f"saved {output}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label="optimized path", linewidth=2)
        ax.scatter(
            constraints[:, 0],
            constraints[:, 1],
            constraints[:, 2],
            label="constraints",
            c=np.linspace(0, 1, constraints.shape[0]),
            cmap="viridis",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
        ax.set_title("traj_opt_py demo")
        image_output = Path(__file__).with_name("demo_result.png")
        fig.tight_layout()
        fig.savefig(image_output, dpi=160)
        print(f"saved {image_output}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
