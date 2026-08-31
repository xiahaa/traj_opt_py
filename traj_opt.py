"""Trajectory optimization helpers for endpoint-based minimum-jerk paths."""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import csc_matrix
from qpsolvers import solve_qp

SEGMENT_DOF = 9
SEGMENT_SPAN = 2 * SEGMENT_DOF
POLY_TERMS = 6
TRAJECTORY_DIM = 3
INEQ_CONSTRAINT_SPAN = 2 * TRAJECTORY_DIM

C = np.block(
    [
        [np.eye(3), np.zeros((3, 15))],
        [np.zeros((3, 9)), np.eye(3), np.zeros((3, 6))],
        [np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))],
        [np.zeros((3, 12)), np.eye(3), np.zeros((3, 3))],
        [np.zeros((3, 6)), np.eye(3), np.zeros((3, 9))],
        [np.zeros((3, 15)), np.eye(3)],
    ]
)

invA = np.array(
    [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 1 / 2, 0, 0, 0],
        [-10, -6, -3 / 2, 10, -4, 1 / 2],
        [15, 8, 3 / 2, -15, 7, -1],
        [-6, -3, -1 / 2, 6, -3, 1 / 2],
    ],
    dtype=float,
)

full_invA = block_diag(invA, invA, invA)

_Q1_TEMPLATE = np.array(
    [
        [10 / 7, 3 / 14, 1 / 84, -10 / 7, 3 / 14, -1 / 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3 / 14, 8 / 35, 1 / 60, -3 / 14, -1 / 70, 1 / 210, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1 / 84, 1 / 60, 1 / 630, -1 / 84, -1 / 210, 1 / 1260, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-10 / 7, -3 / 14, -1 / 84, 10 / 7, -3 / 14, 1 / 84, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3 / 14, -1 / 70, -1 / 210, -3 / 14, 8 / 35, -1 / 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-1 / 84, 1 / 210, 1 / 1260, 1 / 84, -1 / 60, 1 / 630, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 10 / 7, 3 / 14, 1 / 84, -10 / 7, 3 / 14, -1 / 84, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3 / 14, 8 / 35, 1 / 60, -3 / 14, -1 / 70, 1 / 210, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1 / 84, 1 / 60, 1 / 630, -1 / 84, -1 / 210, 1 / 1260, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -10 / 7, -3 / 14, -1 / 84, 10 / 7, -3 / 14, 1 / 84, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3 / 14, -1 / 70, -1 / 210, -3 / 14, 8 / 35, -1 / 60, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1 / 84, 1 / 210, 1 / 1260, 1 / 84, -1 / 60, 1 / 630, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10 / 7, 3 / 14, 1 / 84, -10 / 7, 3 / 14, -1 / 84],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 / 14, 8 / 35, 1 / 60, -3 / 14, -1 / 70, 1 / 210],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / 84, 1 / 60, 1 / 630, -1 / 84, -1 / 210, 1 / 1260],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -10 / 7, -3 / 14, -1 / 84, 10 / 7, -3 / 14, 1 / 84],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 / 14, -1 / 70, -1 / 210, -3 / 14, 8 / 35, -1 / 60],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1 / 84, 1 / 210, 1 / 1260, 1 / 84, -1 / 60, 1 / 630],
    ],
    dtype=float,
)

_Q2_TEMPLATE = np.array(
    [
        [120 / 7, 60 / 7, 3 / 7, -120 / 7, 60 / 7, -3 / 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [60 / 7, 192 / 35, 11 / 35, -60 / 7, 108 / 35, -4 / 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3 / 7, 11 / 35, 3 / 35, -3 / 7, 4 / 35, 1 / 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-120 / 7, -60 / 7, -3 / 7, 120 / 7, -60 / 7, 3 / 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [60 / 7, 108 / 35, 4 / 35, -60 / 7, 192 / 35, -11 / 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [-3 / 7, -4 / 35, 1 / 70, 3 / 7, -11 / 35, 3 / 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 120 / 7, 60 / 7, 3 / 7, -120 / 7, 60 / 7, -3 / 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 60 / 7, 192 / 35, 11 / 35, -60 / 7, 108 / 35, -4 / 35, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3 / 7, 11 / 35, 3 / 35, -3 / 7, 4 / 35, 1 / 70, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -120 / 7, -60 / 7, -3 / 7, 120 / 7, -60 / 7, 3 / 7, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 60 / 7, 108 / 35, 4 / 35, -60 / 7, 192 / 35, -11 / 35, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -3 / 7, -4 / 35, 1 / 70, 3 / 7, -11 / 35, 3 / 35, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120 / 7, 60 / 7, 3 / 7, -120 / 7, 60 / 7, -3 / 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60 / 7, 192 / 35, 11 / 35, -60 / 7, 108 / 35, -4 / 35],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 / 7, 11 / 35, 3 / 35, -3 / 7, 4 / 35, 1 / 70],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -120 / 7, -60 / 7, -3 / 7, 120 / 7, -60 / 7, 3 / 7],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60 / 7, 108 / 35, 4 / 35, -60 / 7, 192 / 35, -11 / 35],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3 / 7, -4 / 35, 1 / 70, 3 / 7, -11 / 35, 3 / 35],
    ],
    dtype=float,
)


def _monomial_basis(t: float) -> np.ndarray:
    return np.array([1.0, t, t**2, t**3, t**4, t**5], dtype=float)


def _basis_block(t: float) -> np.ndarray:
    basis = _monomial_basis(t)
    zero = np.zeros(POLY_TERMS, dtype=float)
    return np.vstack(
        [
            np.concatenate([basis, zero, zero]),
            np.concatenate([zero, basis, zero]),
            np.concatenate([zero, zero, basis]),
        ]
    )


def _segment_index(t_breaks: np.ndarray, value: float) -> int:
    if t_breaks.size < 2:
        return 0
    idx_e = int(np.searchsorted(t_breaks, value, side="right"))
    return int(np.clip(idx_e, 1, t_breaks.size - 1))


def objective_endpoint(x, tau, l, m):
    x = np.asarray(x).reshape(-1)
    num_nodes = x.size // SEGMENT_DOF
    if num_nodes < 2:
        raise ValueError("x must contain at least two endpoint blocks")

    q1 = (_Q1_TEMPLATE / tau) * l
    q2 = (_Q2_TEMPLATE / (tau**3)) * m
    segment_cost = C.T @ (q1 + q2) @ C

    big_q = np.zeros((x.size, x.size), dtype=float)
    for i in range(1, num_nodes):
        sl = slice(i * SEGMENT_DOF - SEGMENT_DOF, i * SEGMENT_DOF + SEGMENT_DOF)
        big_q[sl, sl] += segment_cost

    q = np.zeros(x.size, dtype=float)
    return big_q, q


def eq_constraint_end_pva(x, pva_in):
    if pva_in is None:
        return None, None

    pva = np.asarray(pva_in, dtype=float).reshape(-1)
    mask = ~np.isnan(pva)
    if not np.any(mask):
        return None, None

    idx = np.flatnonzero(mask)
    A = np.zeros((idx.size, np.asarray(x).reshape(-1).size), dtype=float)
    A[np.arange(idx.size), idx] = 1.0
    b = pva[mask]
    return A, b


def eq_pos_constraint_end(x, p_in, t_in, t_s):
    if p_in is None or len(p_in) == 0 or t_in is None or len(t_in) == 0:
        return None, None

    x = np.asarray(x).reshape(-1)
    p_in = np.asarray(p_in, dtype=float).reshape(-1)
    t_in = np.asarray(t_in, dtype=float).reshape(-1)
    t_s = np.asarray(t_s, dtype=float).reshape(-1)

    num_cons = t_in.size
    A = np.zeros((TRAJECTORY_DIM * num_cons, x.size), dtype=float)
    b = np.zeros(TRAJECTORY_DIM * num_cons, dtype=float)

    for i, t in enumerate(t_in):
        idx_e = _segment_index(t_s, float(t))
        idx_s = idx_e - 1
        t_span = t_s[idx_e] - t_s[idx_s]
        basis = _basis_block((t - t_s[idx_s]) / t_span)
        poly = basis @ full_invA @ C
        A[i * 3 : i * 3 + 3, idx_s * SEGMENT_DOF : idx_s * SEGMENT_DOF + SEGMENT_SPAN] = poly
        b[i * 3 : i * 3 + 3] = p_in[i * 3 : i * 3 + 3]

    return A, b


def ineq_pos_constraint_end(x, p_in, t_in, t_s, tol=0.1):
    if p_in is None or len(p_in) == 0 or t_in is None or len(t_in) == 0:
        return None, None

    x = np.asarray(x).reshape(-1)
    p_in = np.asarray(p_in, dtype=float).reshape(-1)
    t_in = np.asarray(t_in, dtype=float).reshape(-1)
    t_s = np.asarray(t_s, dtype=float).reshape(-1)
    tol = np.broadcast_to(np.asarray(tol, dtype=float), p_in.shape)

    num_cons = t_in.size
    G = np.zeros((INEQ_CONSTRAINT_SPAN * num_cons, x.size), dtype=float)
    h = np.zeros(INEQ_CONSTRAINT_SPAN * num_cons, dtype=float)

    for i, t in enumerate(t_in):
        idx_e = _segment_index(t_s, float(t))
        idx_s = idx_e - 1
        t_span = t_s[idx_e] - t_s[idx_s]
        basis = _basis_block((t - t_s[idx_s]) / t_span)
        poly = basis @ full_invA @ C
        sl = slice(idx_s * SEGMENT_DOF, idx_s * SEGMENT_DOF + SEGMENT_SPAN)
        G[i * INEQ_CONSTRAINT_SPAN : i * INEQ_CONSTRAINT_SPAN + TRAJECTORY_DIM, sl] = poly
        G[
            i * INEQ_CONSTRAINT_SPAN + TRAJECTORY_DIM : i * INEQ_CONSTRAINT_SPAN + INEQ_CONSTRAINT_SPAN,
            sl,
        ] = -poly
        h[i * INEQ_CONSTRAINT_SPAN : i * INEQ_CONSTRAINT_SPAN + TRAJECTORY_DIM] = (
            p_in[i * TRAJECTORY_DIM : i * TRAJECTORY_DIM + TRAJECTORY_DIM]
            + tol[i * TRAJECTORY_DIM : i * TRAJECTORY_DIM + TRAJECTORY_DIM]
        )
        h[
            i * INEQ_CONSTRAINT_SPAN + TRAJECTORY_DIM : i * INEQ_CONSTRAINT_SPAN + INEQ_CONSTRAINT_SPAN
        ] = -p_in[i * TRAJECTORY_DIM : i * TRAJECTORY_DIM + TRAJECTORY_DIM] + tol[
            i * TRAJECTORY_DIM : i * TRAJECTORY_DIM + TRAJECTORY_DIM
        ]

    return G, h


def get_polynomial_coefficients(x):
    x = np.asarray(x).reshape(-1)
    num_seg = x.size // SEGMENT_DOF - 1

    poly = {}
    for i in range(num_seg):
        pva = C @ x[i * SEGMENT_DOF : i * SEGMENT_DOF + SEGMENT_SPAN]
        poly[i] = {
            "x": invA @ pva[0:6],
            "y": invA @ pva[6:12],
            "z": invA @ pva[12:18],
        }

    return poly


def get_traj_pts(poly, num_pts_per_seg=200):
    xyz = []
    times = []
    basis_t = np.linspace(0.0, 1.0, num_pts_per_seg)
    basis = np.vstack([_monomial_basis(t) for t in basis_t])

    for i in range(len(poly)):
        coeffs = np.vstack([poly[i]["x"], poly[i]["y"], poly[i]["z"]])
        xyz.append(basis @ coeffs.T)
        times.append(basis_t + i)

    return np.vstack(xyz), np.concatenate(times)


def warp_real_time_to_virtual_time(t_real, t_cons):
    t_real = np.asarray(t_real, dtype=float).reshape(-1)
    t_cons = np.asarray(t_cons, dtype=float).reshape(-1)
    if t_real.size < 2:
        return t_cons.copy()

    idx_e = np.searchsorted(t_real, t_cons, side="right")
    idx_e = np.clip(idx_e, 1, t_real.size - 1)
    idx_s = idx_e - 1
    t_span = t_real[idx_e] - t_real[idx_s]
    return (t_cons - t_real[idx_s]) / t_span + t_real[idx_s]


def optimize(P, q=None, G=None, h=None, A=None, b=None, solver="osqp"):
    P = csc_matrix(P)
    G = None if G is None else csc_matrix(G)
    A = None if A is None else csc_matrix(A)
    x = solve_qp(P, q, G, h, A, b, solver=solver)
    return None if x is None or np.size(x) == 0 else x


def example1(p, t, p_cons, t_cons, l=0, mu=1, tol=0.1):
    x0 = np.zeros((SEGMENT_DOF * len(p),), dtype=float)
    tau = 1.0
    t_s = np.arange(len(p), dtype=float) * tau

    t_in = warp_real_time_to_virtual_time(t, t_cons)
    P, q = objective_endpoint(x0, tau, l, mu)
    A, b = eq_pos_constraint_end(x0, [], [], [])
    G, h = ineq_pos_constraint_end(x0, p_cons, t_in, t_s, tol=tol)
    return optimize(P, q, G, h, A, b)


def naive_uv_constraint(x, uv, tol=0.1):
    if uv is None or len(uv) == 0:
        return None, None

    x = np.asarray(x).reshape(-1)
    uv = np.asarray(uv, dtype=float).reshape(-1)
    if uv.size != 2:
        raise ValueError("uv must contain exactly two values")
    u, v = uv
    tol = float(tol)

    G = np.zeros((4, x.size), dtype=float)
    h = np.zeros(4, dtype=float)

    G[0, 0] = 1
    G[0, 6] = -u - tol
    G[1, 0] = -1
    G[1, 6] = u - tol
    G[2, 3] = 1
    G[2, 6] = -v - tol
    G[3, 3] = -1
    G[3, 6] = v - tol

    return G, h
