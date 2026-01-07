import numpy as np
from scipy.linalg import eigh
import ctypes, os, sys
from grid import build, get_ao_grad
import argparse
from pyscf import gto, dft

libname = {
    'linux': './weights/gga.so',
    'darwin': 'libgga.so',
    'win32': 'dft.dll'
}[sys.platform]
lib = ctypes.CDLL(os.path.abspath(libname))

lib.get_rho_sigma.argtypes = [
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
]
lib.get_rho_sigma.restype = None

lib.build_vxc_matrix_gga.argtypes = [
    ctypes.c_int, ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C')
]
lib.build_vxc_matrix_gga.restype = None

lib.compute_exc_energy_gga.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C')
]
lib.compute_exc_energy_gga.restype = ctypes.c_double

lib.build_coulomb_matrix.argtypes = [
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
]
lib.build_coulomb_matrix.restype = None

def get_rho_grad(dm, ao, ao_grad):
    ngrid, nao = ao.shape
    rho = np.empty(ngrid)
    sigma = np.empty(ngrid)
    grad_rho = np.empty((ngrid, 3))

    lib.get_rho_sigma(
        nao, ngrid,
        np.ascontiguousarray(dm),
        np.ascontiguousarray(ao),
        np.ascontiguousarray(ao_grad),
        rho, sigma, grad_rho
    )

    rho = np.clip(rho, 1e-12, None)
    return rho, sigma, grad_rho


def build_vxc_matrix(dm, ao_values, ao_grad, grids):
    rho, sigma, grad_rho = get_rho_grad(dm, ao_values, ao_grad)
    nao = ao_values.shape[1]
    ngrid = ao_values.shape[0]

    vxc = np.empty((nao, nao))

    lib.build_vxc_matrix_gga(
        nao, ngrid,
        np.ascontiguousarray(ao_values),
        np.ascontiguousarray(ao_grad),
        np.ascontiguousarray(grids.weights),
        np.ascontiguousarray(rho),
        np.ascontiguousarray(sigma),
        np.ascontiguousarray(grad_rho),
        vxc
    )
    return vxc


def compute_exc_energy(dm, ao, ao_grad, grids):
    rho, sigma, _ = get_rho_grad(dm, ao, ao_grad)
    return lib.compute_exc_energy_gga(
        ao.shape[0],
        np.ascontiguousarray(grids.weights),
        np.ascontiguousarray(rho),
        np.ascontiguousarray(sigma)
    )


def build_J(dm, eri):
    nao = dm.shape[0]
    J = np.empty((nao, nao))
    lib.build_coulomb_matrix(
        nao,
        np.ascontiguousarray(eri.reshape(-1)),
        np.ascontiguousarray(dm),
        J
    )
    return J


# HF 交换矩阵（K矩阵）：保留你的直接实现（注意 ERI 排列）
def build_K_matrix(dm, eri):
    """HF exact exchange:K_pq = Σ_rs (pr|qs) D_rs"""
    nao = dm.shape[0]
    K = np.zeros((nao, nao))
    eri4 = eri.reshape(nao, nao, nao, nao)

    for p in range(nao):
        for q in range(nao):
            # sum over r,s: (p r | q s) * D[r,s]
            K[p, q] = np.sum(eri4[p, :, q, :] * dm)  # 改成 p, :, q, : 对应 (p r | q s)
    return K


# ==========================================================
#             正交化对角化（安全版 solve_fock）
# ==========================================================
def orthogonalize_S(S, tol=1e-14):
    """返回 S^{-1/2} 矩阵（用于广义本征问题正交化）"""
    se, sv = eigh(S)
    if np.any(se <= 0):
        raise RuntimeError("Non-positive eigenvalues of S!")
    inv_sqrt = sv @ np.diag(1.0 / np.sqrt(se)) @ sv.T
    return inv_sqrt


def solve_fock_numpy(F, S):
    """标准的 S^{-1/2} F S^{-1/2} diagonalization，
       返回 (eps, C) ，其中 C 是 AO 基底下的 MO 系数（列为轨道），
       并确保 C.T @ S @ C = I"""
    S_m12 = orthogonalize_S(S)
    F_ortho = S_m12 @ F @ S_m12
    e_ortho, C_ortho = eigh(F_ortho)
    # 回到 AO 基底
    C = S_m12 @ C_ortho
    # 归一化/检查：C.T @ S @ C 应该是单位矩阵
    err = np.max(np.abs(C.T @ S @ C - np.eye(C.shape[1])))
    return e_ortho, C, err


# ==========================================================
#                    MP2 与诊断函数（更安全）
# ==========================================================
def diagnose_mp2(e, C, eri, S, nocc, max_print=6):
    nao = C.shape[0]
    print("\n=== MP2 DIAGNOSTICS ===")
    print("nao, nocc:", nao, nocc)
    print("e (first/last 8):", np.array2string(e[:min(8,len(e))], precision=6),
          " ... ", np.array2string(e[-min(8,len(e)):], precision=6))
    ortho = C.T @ S @ C
    err_ortho = np.max(np.abs(ortho - np.eye(nao)))
    print("max|C.T S C - I| =", err_ortho)

    eri4 = eri.reshape(nao, nao, nao, nao)
    print("ERI AO stats: min, max, mean, std =",
          eri4.min(), eri4.max(), eri4.mean(), eri4.std())

    sample = [eri4[idx, idx, idx, idx] for idx in range(min(max_print, nao))]
    print("sample (p,p,p,p):", sample)

    # build some MO integrals to inspect
    try:
        MO = np.einsum('pi,qj,rk,sl,ijkl->pqrs', C, C, C, C, eri4, optimize=True)
        nvir = nao - nocc
        if nocc > 0 and nvir > 0:
            i = 0; j = min(1, nocc-1); a = 0; b = min(1, nvir-1)
            ia = nocc + a; jb = nocc + b
            val = MO[i, ia, j, jb]
            val_asym = val - MO[i, jb, j, ia]
            print(f"MO sample (i={i},a={a},j={j},b={b}): (iajb)={val:.6e}, antisym={val_asym:.6e}")
            denom = e[i] + e[j] - e[ia] - e[jb]
            print("sample denom:", denom)
            # find minimum denom
            min_den = np.inf
            for ii in range(nocc):
                for jj in range(nocc):
                    for aa in range(nvir):
                        for bb in range(nvir):
                            denom = e[ii] + e[jj] - e[nocc+aa] - e[nocc+bb]
                            if abs(denom) < min_den:
                                min_den = abs(denom)
            print("min |denominator| over i,j,a,b =", min_den)
    except Exception as exc:
        print("MO transform failed in diagnostics:", exc)
    print("=== END DIAG ===\n")


def compute_mp2_energy_checked(e, C, eri, S, nocc):
    nao = C.shape[0]
    nvir = nao - nocc
    ortho_err = np.max(np.abs(C.T @ S @ C - np.eye(nao)))
    if ortho_err > 1e-8:
        print("Warning: C not orthonormal w.r.t S (max err=%.3e). Results may be invalid." % ortho_err)

    idx = np.argsort(e)
    if not np.all(idx == np.arange(len(e))):
        e = e[idx].copy()
        C = C[:, idx].copy()

    eri4 = eri.reshape(nao, nao, nao, nao)
    MO = np.einsum('pi,qj,rk,sl,ijkl->pqrs', C, C, C, C, eri4, optimize=True)

    eps_occ = e[:nocc]
    eps_vir = e[nocc:]
    emp2 = 0.0
    min_denom = np.inf
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                for b in range(nvir):
                    ia = nocc + a
                    jb = nocc + b
                    A = MO[i, ia, j, jb]
                    B = MO[i, jb, j, ia]
                    denom = eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b]
                    if abs(denom) < 1e-12:
                        denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12
                        print("Tiny denom detected, adjusted to 1e-12")
                    min_denom = min(min_denom, abs(denom))
                    emp2 += (A * (2.0 * A - B)) / denom
    print("MP2 computed. min |denom| =", min_denom)
    return emp2


# ==========================================================
#                   SCF + B2PLYP 主程序（主修正版）
# ==========================================================
def run_b2plyp(molname, maxiter=50, damping=0.5):
    atom_file = f"./atom_txt/{molname}.txt"
    grid_file = f"./grid_txt/{molname}_grid.txt"
    atom = open(atom_file).read()

    # build 返回：Hcore, S, nocc, T, eri, ao, grids, E_nuc
    Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom, grid_file)
    ao_grad = get_ao_grad(atom, grids)

    nao = Hcore.shape[0]

    # PySCF molecule 用于 PBE 参考（用相同几何和基组）
    mol = gto.M(
        atom=atom,
        basis="cc-pvdz",
        unit="Bohr",
        verbose=0
    )

    # --- 初始 guess: 在正交化基底上 diagonalize Hcore
    S_m12 = orthogonalize_S(S)
    H_ortho = S_m12 @ Hcore @ S_m12
    e_init, C2 = eigh(H_ortho)
    C = S_m12 @ C2  # AO 基底下的 MO 系数
    # 保证列排序与能量一致
    idx = np.argsort(e_init)
    e_init = e_init[idx]
    C = C[:, idx]

    dm = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    E_old = 0.0

    ax = 0.53
    ac = 0.27

    print("\nSCF (B2PLYP-like) started")
    print("-" * 60)

    for it in range(100):
        J = build_J(dm, eri)
        K = build_K_matrix(dm, eri)
        Vxc = build_vxc_matrix(dm, ao_values, ao_grad, grids)

        F = Hcore + J + ax * K + Vxc

        # use safe numpy solve_fock
        e, C, ortho_err = solve_fock_numpy(F, S)
        if ortho_err > 1e-8:
            print(f"Warning: after diagonalization max|C.T S C - I| = {ortho_err:.3e}")

        dm_new = 2.0 * C[:, :nocc] @ C[:, :nocc].T

        E_one = np.einsum("ij,ji->", dm_new, Hcore)
        E_coul = 0.5 * np.einsum("ij,ji->", dm_new, J)
        E_xc_c = compute_exc_energy(dm_new, ao_values, ao_grad, grids)
        E_xc_x = compute_exc_energy(dm_new, ao_values, ao_grad, grids)
        E_x_hf = -0.5 * np.einsum("ij,ji->", dm_new, K)

        E_tot = E_one + E_coul + (1 - ac) * E_xc_c + E_nuc + ax * E_x_hf + (1 - ax) * E_xc_x
        dE = E_tot - E_old

        print(f"{it+1:3d}   Etot = {E_tot:.10f}   dE = {dE:.3e}")

        if abs(dE) < 1e-8:
            break

        # damping/mixing
        dm = damping * dm_new + (1.0 - damping) * dm
        E_old = E_tot

    # SCF 完成，诊断并运行 MP2
    print("SCF done. Running diagnostics for MP2...")
    diagnose_mp2(e, C, eri, S, nocc)

    Emp2 = compute_mp2_energy_checked(e, C, eri, S, nocc)
    E_b2plyp = E_tot + ac * Emp2

    # PySCF PBE 参考
    mf = dft.RKS(mol)
    mf.xc = "PBE"
    E_pbe = mf.kernel()

    print("\n==== B2PLYP Energy ====")
    print(f" SCF energy : {E_tot:.10f}")
    print(f" MP2 corr   : {Emp2:.10f}")
    print(f" Final E    : {E_b2plyp:.10f}")

    print("\n==== PySCF PBE Energy (Reference) ====")
    print(f" PBE Energy : {E_pbe:.10f}")

    print("\n==== Energy Difference ====")
    print(f" B2PLYP - PBE = {E_b2plyp - E_pbe:.10f}")

    return E_b2plyp, E_pbe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mol", type=str)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--damping", type=float, default=0.5)
    args = parser.parse_args()

    run_b2plyp(args.mol, maxiter=args.maxiter, damping=args.damping)
