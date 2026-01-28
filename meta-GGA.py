import numpy as np
from scipy.linalg import eigh
import sys, ctypes, os, time
from datetime import timedelta
from pyscf import gto, dft
# 假设 grid 模块在你本地存在，保持不变
from grid import build, get_ao_grad
import argparse

# --- 加载 C 库 (保持不变) ---
libname = {'linux':'./weights/gga.so',
           'darwin':'libgga.so',
           'win32':'dft.dll'}[sys.platform]
# 为了防止报错，这里假设库文件存在，实际运行时请确保路径正确
try:
    lib = ctypes.CDLL(os.path.abspath(libname))
except OSError:
    print(f"Warning: Could not load {libname}. Computation relying on C-lib will fail.")
    # 创建一个 dummy lib 对象防止 import 阶段报错
    class DummyLib:
        def __getattr__(self, name):
            return lambda *args: None
    lib = DummyLib()

# --- 定义 C 函数参数类型 (保持不变) ---
if hasattr(lib, 'build_vxc_matrix_gga'):
    lib.build_vxc_matrix_gga.argtypes = [
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
    lib.build_vxc_matrix_gga.restype = None

    lib.compute_exc_energy_gga.argtypes = [
        ctypes.c_int,
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
    lib.compute_exc_energy_gga.restype = ctypes.c_double

    lib.build_coulomb_matrix.argtypes = [ctypes.c_int,
                                         np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                         np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                         np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
    lib.build_coulomb_matrix.restype = None

    lib.solve_fock_eigen.argtypes = [ctypes.c_int,
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS'),
                                     np.ctypeslib.ndpointer(ctypes.c_double, flags='C_CONTIGUOUS')]
    lib.solve_fock_eigen.restype = None

    lib.get_rho_sigma.restype = None
    lib.get_rho_sigma.argtypes = [
        ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
        np.ctypeslib.ndpointer(dtype=np.float64, flags='C'),
    ]

# --- 辅助函数 ---

def get_rho_grad(dm, ao_values, ao_grad):
    ngrid, nao = ao_values.shape
    rho   = np.empty(ngrid, dtype=np.float64)
    sigma = np.empty(ngrid, dtype=np.float64)
    grad_rho = np.empty((ngrid, 3), dtype=np.float64) 

    lib.get_rho_sigma(nao, ngrid,
                      np.ascontiguousarray(dm, dtype=np.float64),
                      np.ascontiguousarray(ao_values, dtype=np.float64),
                      np.ascontiguousarray(ao_grad,  dtype=np.float64),
                      rho, sigma, grad_rho)
    
    rho   = np.clip(rho,   1e-12, None)
    return rho, sigma, grad_rho

def compute_tau(dm, ao_grad):
    if ao_grad.shape[0] == 3:
        ao_grad = np.transpose(ao_grad, (1, 0, 2))

    tau = np.einsum(
        'pki,pkj,ij->p',
        ao_grad, ao_grad, dm
    )

    tau = tau * 0.5 
    return tau


def tau_kernel(tau):
    """
    一个简单的 Meta-GGA 示例 kernel (仅用于演示)
    """
    c_tau = 1e-4
    eps_tau = c_tau * tau       # 能量密度
    v_tau   = np.full_like(tau, c_tau) # 对 tau 的导数 dE/dtau
    return eps_tau, v_tau

def build_vxc_tau_python(dm, ao_grad, grids):
    """
    计算动能密度对 Fock 矩阵的贡献 V_xc^tau
    V_ij = integral [ v_tau(r) * d(tau)/d(D_ij) ] dr
    因为 tau = 0.5 * sum(D * grad * grad)
    所以 d(tau)/d(D_ij) = 0.5 * grad_phi_i * grad_phi_j
    """
    if ao_grad.shape[0] == 3:
        ao_grad = np.transpose(ao_grad, (1, 0, 2))

    tau = compute_tau(dm, ao_grad)
    _, v_tau = tau_kernel(tau)

    w = grids.weights

    # 计算 integral [ w * v_tau * (grad_i . grad_j) ]
    # 这里计算的是不带 0.5 的部分
    F_tau = np.einsum(
        'p,pki,pkj->ij',
        w * v_tau,
        ao_grad,
        ao_grad
    )
    
    # 【修正】必须乘以 0.5。
    # 来源：对 tau 定义式求导产生的系数 (d/dD [0.5 * D * ...])
    F_tau *= 0.5 
    
    return F_tau

def build_vxc_matrix(dm, ao_values, ao_grad, grids):
    rho, sigma, grad_rho = get_rho_grad(dm, ao_values, ao_grad)
    nao, ngrid = ao_values.shape[1], ao_values.shape[0]

    ao_c   = np.ascontiguousarray(ao_values,  dtype=np.float64)
    aograd_c = np.ascontiguousarray(ao_grad, dtype=np.float64)
    w_c    = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c  = np.ascontiguousarray(rho,  dtype=np.float64)
    sig_c  = np.ascontiguousarray(sigma, dtype=np.float64)
    grho_c = np.ascontiguousarray(grad_rho, dtype=np.float64)
    vxc_mat = np.empty((nao, nao), dtype=np.float64, order='C')

    lib.build_vxc_matrix_gga(nao, ngrid, ao_c, aograd_c, w_c, rho_c, sig_c, grho_c, vxc_mat)
    return vxc_mat

def compute_exc_energy(dm, ao_values, ao_grad, grids):
    rho, sigma, _ = get_rho_grad(dm, ao_values, ao_grad)
    w_c   = np.ascontiguousarray(grids.weights, dtype=np.float64)
    rho_c = np.ascontiguousarray(rho,  dtype=np.float64)
    sig_c = np.ascontiguousarray(sigma, dtype=np.float64)
    
    # 1. 计算常规 GGA 部分能量
    E_xc_gga = lib.compute_exc_energy_gga(len(grids.coords), w_c, rho_c, sig_c)

    # 2. 计算 Meta-GGA tau 部分能量
    tau = compute_tau(dm, ao_grad)
    eps_tau, _ = tau_kernel(tau)
    E_xc_tau = np.dot(grids.weights, eps_tau)

    return E_xc_gga + E_xc_tau


def build_coulomb_matrix(dm, eri):
    nao = dm.shape[0]
    dm_c = np.ascontiguousarray(dm, dtype=np.float64)
    eri_c = np.ascontiguousarray(eri.reshape(-1), dtype=np.float64)
    J = np.empty((nao, nao), dtype=np.float64, order='C')
    lib.build_coulomb_matrix(nao, eri_c, dm_c, J)
    return J


def solve_fock_equation(F, S):
    n = F.shape[0]
    e = np.empty(n)
    C = np.empty((n, n), order='C')
    lib.solve_fock_eigen(n,
                         np.ascontiguousarray(F, dtype=np.float64),
                         np.ascontiguousarray(S, dtype=np.float64),
                         e, C)
    C = C.reshape(n, n).T
    return e, C


def adaptive_mixing(dm_new, dm_old, cycle, dm_change):
    if cycle < 10:
        mix_param = 0.1
    elif dm_change > 1e-3:
        mix_param = 0.2
    elif dm_change > 1e-4:
        mix_param = 0.3
    else:
        mix_param = 0.5
    return mix_param * dm_new + (1 - mix_param) * dm_old


# --- Main 程序 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GGA calculation for a given molecule.")
    parser.add_argument("molecule", type=str, help="Name of the molecule (e.g., h2o, dha)")
    args = parser.parse_args()

    atom = args.molecule.lower()
    try:
        # 这里假设当前目录下有 atom_txt 文件夹
        with open(f"./atom_txt/{atom}.txt", "r") as f:
            atom_structure = f.read()
    except FileNotFoundError:
        print(f"Error: No structure found for molecule {atom}.")
        exit(1)
    
    # 这里假设 grid_txt 文件夹和 grid.py 模块可用
    grid_add = f"./grid_txt/{atom}_grid.txt"
    try:
        Hcore, S, nocc, T, eri, ao_values, grids, E_nuc = build(atom_structure, grid_add)
        ao_grad = get_ao_grad(atom_structure, grids)
    except Exception as e:
        print(f"Error during grid/molecule build: {e}")
        exit(1)

    e_init, C_init = eigh(Hcore, S)
    dm = 2 * C_init[:, :nocc] @ C_init[:, :nocc].T
    start_time = time.time()
    print("\nSCF started!")
    print("-" * 70)
    print(f"{'epoch':>4} {'tot energy':>15} {'Δenergy':>12} {'Δdensity':>12}")
    print("-" * 70)

    converged = False
    E_old = 0.0
    Vxc_time, Exc_time = [], []

    for cycle in range(100):
        J = build_coulomb_matrix(dm, eri)

        t1 = time.time()
        Vxc = build_vxc_matrix(dm, ao_values, ao_grad, grids)
        
        # 计算 tau 相关的势 (包含修正后的 0.5 因子)
        Vxc_tau = build_vxc_tau_python(dm, ao_grad, grids)
        
        t2 = time.time()
        Vxc_time.append(t2 - t1)
        
        F = Hcore + J + Vxc + Vxc_tau
        e, C = solve_fock_equation(F, S)
        dm_new = 2 * C[:, :nocc] @ C[:, :nocc].T

        E_one  = np.einsum('ij,ji->', dm_new, Hcore)
        E_coul = 0.5 * np.einsum('ij,ji->', dm_new, J)

        t3 = time.time()
        # 这里的 compute_exc_energy 内部已经包含了修正后的 tau 能量计算
        E_xc = compute_exc_energy(dm_new, ao_values, ao_grad, grids)

        t4 = time.time()
        Exc_time.append(t4 - t3)

        E_tot = E_one + E_coul + E_xc + E_nuc
        dE    = E_tot - E_old
        dm_change = np.linalg.norm(dm_new - dm)

        print(f"{cycle+1:4d} {E_tot:15.8f} {dE:12.6e} {dm_change:12.6e}")

        if abs(dE) < 1e-8 and dm_change < 1e-6:
            converged = True
            end_time = time.time()
            print(f"SCF converged! E = {E_tot:.8f} Hartree")
            print(f' E_one  : {E_one:.6f} Hartree')
            print(f' E_coul : {E_coul:.6f} Hartree')
            print(f' E_exc  : {E_xc:.6f} Hartree')
            print(f"Vxc_average_time: {sum(Vxc_time)/len(Vxc_time)*1000:.6f} ms")
            print(f"Exc_average_time: {sum(Exc_time)/len(Exc_time)*1000:.6f} ms")
            print(f"Total time: {end_time - start_time:.6f} s\n")
            break

        dm = adaptive_mixing(dm_new, dm, cycle, dm_change)
        E_old = E_tot

    if not converged:
        print("Warning: SCF unconverged!")

    # PySCF 参考部分保持不变...
    try:
        mol = gto.Mole()
        mol.atom = atom_structure
        mol.basis = 'sto-3g'
        mol.build()

        mf = dft.RKS(mol)
        # 1. 这里改为标准的 Meta-GGA 泛函名称，如 'TPSS', 'SCAN', 'M06-L'
        mf.xc = 'TPSS' 
        # 2. Meta-GGA 通常需要更精细的积分格点（可选，但推荐）
        mf.grids.level = 4 

        mf.kernel()

        # 获取能量成分
        dm_ref = mf.make_rdm1()
        h1 = mol.intor('int1e_kin') + mol.intor('int1e_nuc')
        E1 = np.einsum('ij,ji->', h1, dm_ref)
        vh = mf.get_j(mol, dm_ref)
        Ecoul = 0.5 * np.einsum('ij,ji->', vh, dm_ref)

        # 提取 XC 能量
        # 注意：mf.energy_elec() 返回 (E_elec, E_coul + E_exc)
        # 所以 E_exc = E_elec - E1 - Ecoul
        Exc_ref = mf.energy_elec()[0] - (E1 + Ecoul)
        Etot_ref = mf.energy_tot()

        print('PySCF (TPSS) reference:')
        print(f' E_one  : {E1:.6f} Hartree')
        print(f' E_coul : {Ecoul:.6f} Hartree')
        print(f' E_exc  : {Exc_ref:.6f} Hartree (TPSS)')
        print(f' E_tot  : {Etot_ref:.8f} Hartree')
    except Exception as e:
        print(f"PySCF reference calculation failed or not installed: {e}")