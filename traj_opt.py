import numpy as np
from scipy.linalg import block_diag
from qpsolvers import solve_qp
import qpsolvers

C = np.block([
            [np.eye((3)), np.zeros((3,15))], 
            [np.zeros((3,9)), np.eye((3)), np.zeros((3,6))],
            [np.zeros((3,3)), np.eye((3)), np.zeros((3,12))],
            [np.zeros((3,12)), np.eye((3)), np.zeros((3,3))],
            [np.zeros((3,6)), np.eye((3)), np.zeros((3,9))], 
            [np.zeros((3,15)), np.eye((3))]
            ])

invA = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0],
    [0,0,1/2,0,0,0],
    [-10,-6,-3/2,10,-4,1/2],
    [15,8,3/2,-15,7,-1],
    [-6,-3,-1/2,6,-3,1/2]
])

full_invA = block_diag(invA,invA,invA)

def objective_endpoint(x, tau, l, m):
    Q1 = np.array([
        [  10/7,  3/14,   1/84, -10/7,   3/14,  -1/84,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [  3/14,  8/35,   1/60, -3/14,  -1/70,  1/210,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [  1/84,  1/60,  1/630, -1/84, -1/210, 1/1260,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [ -10/7, -3/14,  -1/84,  10/7,  -3/14,   1/84,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [  3/14, -1/70, -1/210, -3/14,   8/35,  -1/60,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [ -1/84, 1/210, 1/1260,  1/84,  -1/60,  1/630,     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0,  10/7,  3/14,   1/84, -10/7,   3/14,  -1/84,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0,  3/14,  8/35,   1/60, -3/14,  -1/70,  1/210,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0,  1/84,  1/60,  1/630, -1/84, -1/210, 1/1260,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0, -10/7, -3/14,  -1/84,  10/7,  -3/14,   1/84,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0,  3/14, -1/70, -1/210, -3/14,   8/35,  -1/60,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0, -1/84, 1/210, 1/1260,  1/84,  -1/60,  1/630,     0,     0,      0,     0,      0,      0],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0,  10/7,  3/14,   1/84, -10/7,   3/14,  -1/84],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0,  3/14,  8/35,   1/60, -3/14,  -1/70,  1/210],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0,  1/84,  1/60,  1/630, -1/84, -1/210, 1/1260],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0, -10/7, -3/14,  -1/84,  10/7,  -3/14,   1/84],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0,  3/14, -1/70, -1/210, -3/14,   8/35,  -1/60],
        [     0,     0,      0,     0,      0,      0,     0,     0,      0,     0,      0,      0, -1/84, 1/210, 1/1260,  1/84,  -1/60,  1/630]])

    Q2 = np.array([
        [  120/7,   60/7,   3/7, -120/7,   60/7,   -3/7,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [   60/7, 192/35, 11/35,  -60/7, 108/35,  -4/35,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [    3/7,  11/35,  3/35,   -3/7,   4/35,   1/70,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [ -120/7,  -60/7,  -3/7,  120/7,  -60/7,    3/7,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [   60/7, 108/35,  4/35,  -60/7, 192/35, -11/35,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [   -3/7,  -4/35,  1/70,    3/7, -11/35,   3/35,      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,  120/7,   60/7,   3/7, -120/7,   60/7,   -3/7,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,   60/7, 192/35, 11/35,  -60/7, 108/35,  -4/35,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,    3/7,  11/35,  3/35,   -3/7,   4/35,   1/70,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0, -120/7,  -60/7,  -3/7,  120/7,  -60/7,    3/7,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,   60/7, 108/35,  4/35,  -60/7, 192/35, -11/35,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,   -3/7,  -4/35,  1/70,    3/7, -11/35,   3/35,      0,      0,     0,      0,      0,      0], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,  120/7,   60/7,   3/7, -120/7,   60/7,   -3/7], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,   60/7, 192/35, 11/35,  -60/7, 108/35,  -4/35], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,    3/7,  11/35,  3/35,   -3/7,   4/35,   1/70], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0, -120/7,  -60/7,  -3/7,  120/7,  -60/7,    3/7], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,   60/7, 108/35,  4/35,  -60/7, 192/35, -11/35], 
        [      0,      0,     0,      0,      0,      0,      0,      0,     0,      0,      0,      0,   -3/7,  -4/35,  1/70,    3/7, -11/35,   3/35]])
    Q1 = Q1 / tau * l
    Q2 = Q2 / (tau ** 3) * m

    N = round(len(x)/9)

    BigQ = np.zeros((9*N,9*N))

    # 18x18 18x18 18x18 = 18x18
    CtQ1C = C.T@Q1@C
    CtQ2C = C.T@Q2@C

    # print(CtQ2C)

    for i in range(1,N):
        BigQ[i*9-9:i*9+9,i*9-9:i*9+9] = BigQ[i*9-9:i*9+9,i*9-9:i*9+9] + CtQ1C + CtQ2C

    ## data terms
    # Q3 = np.zeros(BigQ.shape)
    fg = np.zeros((BigQ.shape[0],1)).reshape(BigQ.shape[0],)

    Q = BigQ.copy() # + Q3;
    Q = (Q+Q.T)

    return (Q, fg)
          

def eq_constraint_end_pva(x, pva_in):
    if len(pva_in) is 0 or pva_in is None:
        A = None
        b = None
    else:
        num_cons = len(pva_in) - np.sum(np.isnan(pva_in))
        if num_cons == 0:
            A = None
            b = None
        else:
            A = np.zeros((num_cons, len(x)))
            b = np.zeros((num_cons,))
            j = 0
            for i, val in enumerate(pva_in):
                if np.isnan(val) != True:
                    A[j, i] = 1
                    b[j] = val
                    j+=1;
    return A, b      


def eq_pos_constraint_end(x, p_in, t_in, t_s):
    # if use endpoint, there is no equality constraints
    if len(p_in) is 0:
        A = None
        b = None
    else:
        num_cons = len(t_in)
        A = np.zeros((3*num_cons, len(x)))
        b = np.zeros((3*num_cons,))

        for i in range(0,num_cons):
            try:
                idx_e = np.where(t_s > t_in[i])[0][0]
            except:
                idx_e = t_s.shape[0] - 1
            idx_s = idx_e - 1
            
            t_span = t_s[idx_e] - t_s[idx_s]
            t = ( t_in[i] - t_s[idx_s] ) / t_span ## map t to [0,1]
            tt = np.array([1,t, t**2, t**3, t**4, t**5])
            full_tt = np.block([[tt, np.zeros(1,12)],[np.zeros(1,6), tt, np.zeros(1,6)],[np.zeros(1,12), tt]]) # 3 x 18
            poly = full_tt@full_invA@C # 3x18 x 18x18 = 3 x 18

            A[i*3:i*3+3,idx_s*9:idx_s*9+18] = poly
            b[i*3:i*3+3] = p_in[i*3:i*3+3]
            #pva = C@x[idx_s*9:idx_s*9+18]
            #poly_x = invA@pva[0:6]
            #poly_y = invA@pva[6:12]
            #poly_z = invA@pva[12:18]

    return (A,b)

def ineq_pos_constraint_end(x, p_in, t_in, t_s, tol=0.1):
    # if use endpoint, there is no equality constraints
    if len(p_in) is 0:
        G = None
        h = None
    else:
        num_cons = len(t_in)
        print("num_cons: %d"%num_cons)
        G = np.zeros((6*num_cons, len(x)))
        h = np.zeros((6*num_cons,))

        if isinstance(tol,np.ndarray) is False:
            tol = np.ones(p_in.shape)*tol
        print(tol)
        # full_invA = block_diag(invA,invA,invA)

        for i in range(0,num_cons):
            # v1 = p_in[i*3:i*3+3]
            try:
                idx_e = np.where(t_s > t_in[i])[0][0]
            except:
                idx_e = t_s.shape[0] - 1
            idx_s = idx_e - 1
            # idx_s = np.where(t_s <= t_in[i])[0][-1]
            # print(idx_e)
            # print(idx_s)
            # assert(0)
            t_span = t_s[idx_e] - t_s[idx_s]
            t = ( t_in[i] - t_s[idx_s] ) / t_span ## map t to [0,1]
            tt = np.array([1,t, t**2, t**3, t**4, t**5])
            full_tt = np.block([[tt, np.zeros((1,12))],[np.zeros((1,6)), tt, np.zeros((1,6))],[np.zeros((1,12)), tt]]) # 3 x 18
            poly = full_tt@full_invA@C # 3x18 x 18x18 = 3 x 18

            G[i*6:i*6+3,idx_s*9:idx_s*9+18] = poly
            G[i*6+3:i*6+6,idx_s*9:idx_s*9+18] = -poly

            h[i*6:i*6+3] = p_in[i*3:i*3+3] + tol[i*3:i*3+3]
            h[i*6+3:i*6+6] = -p_in[i*3:i*3+3] + tol[i*3:i*3+3]
        
    # P0 = np.array([[1,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,1,0,0]])
    # N = round(len(x0)/9)
    # G = np.zeros((6*len(indices),len(x0)))
    # h = np.zeros((6*len(indices),1)).squeeze()
    # tol = 0.01
    # for i, ii in enumerate(indices):
        # v1 = p[i*3:i*3+3]
        # G[i*6:i*6+3, ii*9:ii*9+9] = P0
        # G[i*6+3:i*6+6, ii*9:ii*9+9] = -P0

        # h[i*6:i*6+3] = v1 + tol
        # h[i*6+3:i*6+6] = -v1 + tol
    return G, h

def get_polynomial_coefficients(x):
    num_seg = len(x)//9-1

    poly = dict()
    for i in range(0,num_seg):
        pva = C@x[i*9:i*9+18]
        # print(pva)

        poly_x = invA@pva[0:6]
        poly_y = invA@pva[6:12]
        poly_z = invA@pva[12:18]
        # print(poly_x)
        poly[i] = {'x':poly_x,'y':poly_y,'z':poly_z}

    return poly


def get_traj_pts(poly, num_pts_per_seg = 200):
    xyz = []
    times = []
    num_seg = len(poly.keys())
    for i in range(0,num_seg):
        poly_x = poly[i]['x'];poly_y = poly[i]['y'];poly_z = poly[i]['z']
        for t in np.linspace(0,1,num_pts_per_seg):
            tt = np.array([1,t, t**2, t**3, t**4, t**5])
            sx = np.dot(poly_x,tt)
            sy = np.dot(poly_y,tt)
            sz = np.dot(poly_z,tt)
            xyz.append([sx,sy,sz])
            times.append(t + i)
    xyz = np.array(xyz)
    # t = np.linspace(0,num_seg,200*num_seg)
    times = np.array(times)
    return (xyz, times)


def warp_real_time_to_virtual_time(t_real, t_cons):
    t_v = []
    for t_c in t_cons:
        try:
            idx_e = np.where(t_real > t_c)[0][0]
        except:
            # print(t.shape)
            idx_e = t_real.shape[0] - 1
        idx_s = idx_e - 1
        t_span = t_real[idx_e] - t_real[idx_s]
        t = ( t_c - t_real[idx_s] ) / t_span + t_real[idx_s] ## map t to [0,1]
        t_v.append(t)
    t_v = np.array(t_v)
    return t_v

def optimize(P, q = None, G = None, h = None,A = None,b = None):
    x = solve_qp(P, q, G, h, A, b)
    if x is None or len(x) is 0:
        return None
    else:
        return x

def example1(p, t, p_cons, t_cons, l = 0, mu = 1, tol = 0.1):
    x0 = np.zeros((9*p.shape[0],1))
    tau = 1
    # l = 0
    # mu = 1
    t_s = np.arange(0,p.shape[0])*tau

    t_in = warp_real_time_to_virtual_time(t, t_cons)
    print(t_in)
    # t_in = np.array([0,0.5,1,1.3,2])
    P, q = objective_endpoint(x0, tau, l, mu)
    A, b = eq_pos_constraint_end(x0,[],[],[])
    G, h = ineq_pos_constraint_end(x0, p_cons, t_in, t_s, tol=tol)
    # print(P.shape)
    # print(h)
    x = optimize(P, q, G, h, A, b)
    return x


def naive_uv_constraint(x, uv, tol=0.1):
    if uv is None or len(uv) is 0:
        G = None
        h = None
    else:
        u,v = uv

        G = np.zeros((4,len(x)))
        h = np.zeros((4,))

        G[0,0] = 1
        G[0,6] = -u - tol

        G[1,0] = -1
        G[1,6] =  u - tol

        G[2,3] = 1
        G[2,6] = -v - tol

        G[3,3] = -1
        G[3,6] =  v - tol
    
    return (G, h)
