import numpy as np
import sys
import os
import warnings

warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis import pca, align
except ImportError:
    print("Error: 请先安装 MDAnalysis (pip install MDAnalysis)")
    sys.exit(1)

# ==========================================
# 1. 论文级视觉控制参数 (重点修改区)
# ==========================================
PDB_FILE = "first_frame_backbone.pdb"
TRAJ_FILE = "every50th_frame.xtc"

STATES = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
FRAMES_PER_STATE = 2000

# --- 净化画面的核心参数 ---
MIN_SHIFT = 3.0         # 物理阈值(Å)：真实物理位移小于 3.0 埃的残基，直接忽略！(过滤微小扭动)
SPARSE_STEP = 4         # 空间稀疏化：每隔 4 个氨基酸才画 1 个箭头 (消除毛毛虫效应，极其关键)

SCALE_FACTOR = 2.5      # 视觉放大倍数：为了让剩下的箭头更有视觉冲击力，可以适当拉长
CYL_RADIUS = 0.5        # 箭身加粗
CONE_RADIUS = 1.0       # 箭头加粗
ARROW_COLOR = "red"     # 统一使用红色，显得更有力量感

# ==========================================
# 2. 加载与对齐
# ==========================================
print(f"正在加载系统并执行全局对齐...")
u = mda.Universe(PDB_FILE, TRAJ_FILE)
ca = u.select_atoms("name CA")
backbone = u.select_atoms("backbone")

aligner = align.AlignTraj(u, u, select='name CA', in_memory=True).run()

# ==========================================
# 3. 核心计算循环
# ==========================================
for i in range(len(STATES) - 1):
    state_A = STATES[i]
    state_B = STATES[i+1]
    
    start_frame = i * FRAMES_PER_STATE
    mid_frame = (i + 1) * FRAMES_PER_STATE
    end_frame = (i + 2) * FRAMES_PER_STATE
    
    transition_name = f"{state_A}_to_{state_B}"
    print(f"\n开始分析: {transition_name}")
    
    # 计算真实物理位移方向用于校正
    coords_A = [ca.positions.copy() for _ in u.trajectory[start_frame:mid_frame]]
    mean_A = np.mean(coords_A, axis=0)
    
    coords_B = [ca.positions.copy() for _ in u.trajectory[mid_frame:end_frame]]
    mean_B = np.mean(coords_B, axis=0)
    real_displacement = mean_B - mean_A 
    
    # 运行 PCA
    pc = pca.PCA(u, select='name CA', align=False)
    pc.run(start=start_frame, stop=end_frame)
    
    pc1_vector = pc.p_components[:, 0].reshape(-1, 3)
    amplitude = np.sqrt(pc.variance[0])
    
    # 方向校正
    dot_product = np.sum(real_displacement * pc1_vector)
    if dot_product < 0:
        pc1_vector = -pc1_vector
        
    print(f"  -> PC1 解释方差: {pc.cumulated_variance[0]*100:.1f}%")
    
    # 生成完美主链底图 PDB
    mean_bb_coords = np.zeros_like(backbone.positions)
    for ts in u.trajectory[start_frame:end_frame]:
        mean_bb_coords += backbone.positions
    mean_bb_coords /= (end_frame - start_frame)
    backbone.positions = mean_bb_coords
    pdb_out = f"avg_backbone_{transition_name}.pdb"
    backbone.write(pdb_out)
    
    # --- 步骤 D: 生成稀疏化的 VMD 脚本 ---
    tcl_out = f"porcupine_{transition_name}.tcl"
    arrow_count = 0
    
    with open(tcl_out, 'w') as f:
        f.write(f"# VMD Porcupine Plot for {transition_name}\n")
        f.write(f"proc draw_porcupine_{state_A}_{state_B} {{}} {{\n")
        f.write("    graphics top material Opaque\n")
        f.write(f"    graphics top color {ARROW_COLOR}\n")
        
        ca_mean_coords = pc.mean.reshape(-1, 3) 
        
        # 【关键修改】：使用 SPARSE_STEP 跳过氨基酸，进行空间抽样
        for atom_idx in range(0, len(ca_mean_coords), SPARSE_STEP):
            start_xyz = ca_mean_coords[atom_idx]
            
            # 计算真实的物理运动距离 (不受 SCALE_FACTOR 影响)
            physical_shift = np.linalg.norm(pc1_vector[atom_idx] * amplitude)
            
            # 物理阈值过滤：剔除小于 3.0 埃的假阳性位移
            if physical_shift < MIN_SHIFT:
                continue
                
            # 计算视觉渲染用的放大向量
            visual_vec = pc1_vector[atom_idx] * amplitude * SCALE_FACTOR
            end_xyz = start_xyz + visual_vec
            
            # 计算箭头尖端
            visual_distance = np.linalg.norm(visual_vec)
            direction = visual_vec / visual_distance
            tip_xyz = end_xyz + direction * (CONE_RADIUS * 2.5) # 尖端比例自适应
            
            # 写入画图指令
            f.write(f"    graphics top cylinder {{{start_xyz[0]:.3f} {start_xyz[1]:.3f} {start_xyz[2]:.3f}}} "
                    f"{{{end_xyz[0]:.3f} {end_xyz[1]:.3f} {end_xyz[2]:.3f}}} radius {CYL_RADIUS} resolution 20\n")
            f.write(f"    graphics top cone {{{end_xyz[0]:.3f} {end_xyz[1]:.3f} {end_xyz[2]:.3f}}} "
                    f"{{{tip_xyz[0]:.3f} {tip_xyz[1]:.3f} {tip_xyz[2]:.3f}}} radius {CONE_RADIUS} resolution 20\n")
            
            arrow_count += 1
            
        f.write("}\n")
        f.write(f"draw_porcupine_{state_A}_{state_B}\n")
        
    print(f"  -> 净化完成！仅生成了 {arrow_count} 根代表性箭头。")

print("\n所有画面已按顶刊标准优化。")
