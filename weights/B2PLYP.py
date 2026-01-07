import numpy as np
from pyscf import gto, dft, scf, mp

# ======== B2PLYP 参数 =========
a_x = 0.53     # HF 交换系数
a_c = 0.27     # MP2 相关系数

# ======== 分子定义 =========
mol = gto.Mole()
mol.atom = '''
O 0.000000  0.000000  0.000000
H 0.758602  0.000000  0.504284
H -0.758602  0.000000  0.504284
'''
mol.basis = "def2-svp"
mol.spin = 0
mol.build()

# ======== 1. 使用 PySCF 原生 B2PLYP 作参考 =========
mf_pyscf = dft.RKS(mol)
mf_pyscf.xc = "b2plyp"
mf_pyscf.kernel()               # 自洽场能量（包含 DFA + HF 部分）
mp2_ref = mp.MP2(mf_pyscf).kernel()[0]  # PySCF 的 MP2 相关能
E_pyscf_b2plyp = mf_pyscf.e_tot + a_c * mp2_ref   # PySCF 的总能量


# ======== 2. 自己手动构造 B2PLYP = DFA + HF + MP2 =========

# --- 2.1 DFA（GGA 如 PBE） ---
mf_dfa = dft.RKS(mol)
mf_dfa.xc = "pbe"         # 你可以换成 lda、blyp 等
mf_dfa.kernel()
E_x_dfa, E_c_dfa = mf_dfa.energy_elec()[1]   # PySCF 返回 (Ex, Ec)

# --- 2.2 HF 交换 ---
mf_hf = scf.RHF(mol)
mf_hf.kernel()
E_x_hf = mf_hf.energy_elec()[1][0]    # HF exchange = (Ex,0)

# --- 2.3 MP2 ---
mp2_obj = mp.MP2(mf_hf)
emp2, t2 = mp2_obj.kernel()

# ---- 2.4 组装 B2PLYP 能量 ----
E_b2plyp_manual = (a_x * E_x_hf
                   + (1 - a_x) * E_x_dfa
                   + (1 - a_c) * E_c_dfa
                   + a_c * emp2
                   + mf_hf.energy_nuc())


# ======== 3. 打印对比 =========
print("\n================ 结果对比 ================\n")
print(f"PySCF B2PLYP 总能量        : {E_pyscf_b2plyp:16.10f}")
print(f"手工构造 B2PLYP 总能量     : {E_b2plyp_manual:16.10f}")
print(f"二者差值                    : {E_b2plyp_manual - E_pyscf_b2plyp:16.5e}")
print("\n==========================================")
