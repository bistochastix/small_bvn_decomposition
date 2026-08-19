import random
from scipy.optimize import linear_sum_assignment

def random_bistochastix(n):
    u = []
    for _ in range(factorial(n)):
        u.append(abs(QQ.random_element(distribution="1/n")))
    u = vector(u)
    u /= sum(u)

    B = 0
    P = Permutation(range(1, n + 1))
    i = 0
    while P:
        B += u[i] * P.to_matrix()
        P = P.next()
        i += 1

    return B, u

def get_permutation_basis(n):
    dim_vector_space = factorial(n)
    basis_vectors = []
    P = Permutation(range(1, n + 1))
    for _ in range(dim_vector_space):
        basis_vectors.append(vector(P.to_matrix()))
        P = P.next()
    return column_matrix(basis_vectors)

def normalize_bistoch(M):
    for i in range(M.nrows()):
        M[i] = [e / sum(M[i]) for e in M[i]]

def verify_decomposition(a, A):
    n = A.nrows()
    A_ = recompose(a, n)
    normalize_bistoch(A_)
    normalize_bistoch(A)
    return A_ == A

def recompose(coeffs, n):
    A = 0
    i = 0
    P = Permutation([e + 1 for e in range(n)])
    while P:
        a_i = coeffs[i]
        A += a_i * P.to_matrix()
        i += 1
        P = P.next()

    return A

def orthogonal_projector(n):
    N = factorial(n)
    S = get_permutation_basis(n)
    B = S.right_kernel_matrix()
    G, _ = B.gram_schmidt()
    K = (G * G.T).apply_map(sqrt)^-1 * G
    L = K.T * K
    assert L == B.T * (B * B.T)^-1 * B
    I = identity_matrix(N)

    return (I - L)

def compute_alpha_k(p_k, a_k, W_k, N):
    """
    Helper function for `active_set_bvn_decompose`.
    """
    alpha_k = 1
    index = -1
    for i in range(N):
        if i in W_k or p_k[i] >= 0:
            continue

        alpha_i = -a_k[i] / p_k[i]
        if alpha_i < alpha_k:
            alpha_k = alpha_i
            index = i
    return min(1, alpha_k), index

def active_set_bvn_decompose(A, a):
    """
    Solve:
        min ||a||^2
        s.t. S*a = Φ(A)
             sum(a) = 1
             a >= 0

    Returns:
        the smallest BvN decomposition of A.
    """
    n = A.nrows()
    N = factorial(n)
    S = get_permutation_basis(n)

    phi_A = A.coefficients()
    I = identity_matrix(N)
    _1 = vector(QQ, [1] * N)
    KKT = block_matrix(QQ, [
        [2 * I, -S.T, matrix(_1).T],
        [S, zero_matrix(n * n, n * n), zero_matrix(n * n, 1)],
        [matrix(_1), zero_matrix(1, n * n), zero_matrix(1, 1)],
    ])

    k = 0
    a_k = a
    W_k = {i for i in range(N) if a_k[i] == 0}
    while True:
        g_k = 2 * a_k
        H_k = I.matrix_from_rows(sorted(W_k))
        c = len(W_k)

        # [ 2I   -S^T   -1 -H_k^T ]
        # [ S      0     0      0 ]
        # [ 1      0     0      0 ]
        # [ H_k    0     0      0 ]
        KKT_stack = block_matrix(QQ, [[H_k, zero_matrix(c, n * n), zero_matrix(c, 1)]])
        KKT_augment = block_matrix(QQ, [[-H_k.T], [zero_matrix(n * n, c)], [zero_matrix(1, c)], [zero_matrix(c, c)]])
        KKT_subPQ = KKT.stack(KKT_stack).augment(KKT_augment)

        rhs = (-g_k).concatenate(vector(QQ, n * n + 1 + c))
        sol = KKT_subPQ.solve_right(rhs)
        p_k = sol[:N]
        nu_k = sol[-c:]

        if p_k != 0:
            alpha_k, idx = compute_alpha_k(p_k, a_k, W_k, N)
            if alpha_k < 1:
                W_k.add(idx)
            a_k = a_k + alpha_k * p_k
        else:
            removed = False
            for j, mul in zip(sorted(W_k), nu_k):
                if mul < 0:
                    W_k.remove(j)
                    removed = True
                    break
            if not removed:
                return a_k

def to_float_matrix(M, prec=4):
    """
    Converts entries of the matrix M from exact ring QQ to float with precision
    `prec`. Used for printing M.
    """
    M_ = matrix(RDF, M.nrows(), M.ncols())
    for i in range(M.nrows()):
        for j in range(M.ncols()):
            M_[i, j] = round(float(M[i, j]), prec)
    return M_

def k(A, minimize=True):
    """
    Minimum weight assignment problem solved with hungarian algorithm.
    """
    _, P = linear_sum_assignment(A) if minimize else linear_sum_assignment(-A)
    h = 0
    for i, j in enumerate(P):
        h += A[i, j]
    return h


n = 5
Q = orthogonal_projector(n)
A, a = random_bistochastix(n)
u = Q * a
# Loop until we generate a random bistochastic matrix which orthogonal
# projection is the smallest BvN decomp. Compare it against the decomp. obtained
# by solving the QP.
while any(e < 0 for e in u):
    A, a = random_bistochastix(n)
    u = Q * a
u_qp = active_set_bvn_decompose(A, a)

assert verify_decomposition(u, A)
assert verify_decomposition(u_qp, A)
assert u == u_qp

A, a = random_bistochastix(n)
u_qp = active_set_bvn_decompose(A, a)
assert verify_decomposition(u_qp, A)
