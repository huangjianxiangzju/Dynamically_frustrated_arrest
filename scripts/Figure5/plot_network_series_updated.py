import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.patches as mpatches
import re
import argparse
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 0. 命令行参数与配置
# ==========================================
parser = argparse.ArgumentParser(description="Plot Domain-level Dynamic Interaction Network in 2 Rows")
parser.add_argument("mode", choices=['saltbridge', 'hbond', 'hydro'], help="Interaction type")
parser.add_argument("--cutoff", type=float, default=None)
args = parser.parse_args()

MODE = args.mode
if MODE == 'saltbridge':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_salt_bridges.csv', 'SB', 'Salt Bridges'
    CUTOFF = args.cutoff if args.cutoff is not None else 0.50
elif MODE == 'hbond':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_hbonds.csv', 'Hbond', 'Hydrogen Bonds'
    CUTOFF = args.cutoff if args.cutoff is not None else 0.30
elif MODE == 'hydro':
    FILE_NAME, PREFIX, TITLE_NAME = 'competitive_hydrophobic.csv', 'Hydro', 'Hydrophobic Contacts'
    CUTOFF = args.cutoff if args.cutoff is not None else 0.40

states = ['6nt', '8nt', '10nt', '12nt', '14nt', '16nt', '18nt']
display_states = [s.replace('nt', '-nt') for s in states]

# ==========================================
# 1. 结构域映射字典与高级扁平化配色方案
# ==========================================
domain_defs = {
    'RuvC': [(1, 59), (718, 764), (919, 1100)],
    'BH': [(60, 94)],
    'REC1': [(95, 176), (306, 495)],
    'REC2': [(177, 305)],
    'REC3': [(496, 717)],
    'L1': [(765, 780)],
    'HNH': [(781, 905)],
    'L2': [(906, 918)],
    'PI': [(1101, 1368)],
}

domain_colors = {
    'RuvC': '#337ACC',  
    'BH': '#FF33CC',    
    'REC1': '#E6E6E5',  
    'REC2': '#DAD7EB',  
    'REC3': '#FCD1A6',  
    'L1': '#C8C8C8',    
    'HNH': '#FFFF7A',   
    'L2': '#C8C8C8',    
    'PI': '#FF9999',    
    'sgRNA': (0.96, 0.72, 0),    
    'TS': (0, 0, 1),             
    'NTS': (0.98, 0, 0.23),      
    'Other': '#B15928'
}

domain_order = ['RuvC', 'BH', 'REC1', 'REC2', 'REC3', 'L1', 'HNH', 'L2', 'PI', 'sgRNA', 'TS', 'NTS']

def get_domain(res_str):
    """
    极速结构域判定：直接利用底层已经清洗好的前缀 (sgRNA, TS, NTS)
    """
    res_str = str(res_str)
    
    # 1. 核酸部分：直接通过底层传递过来的标准前缀判定
    if res_str.startswith('sgRNA'): return 'sgRNA'
    if res_str.startswith('TS'): return 'TS'
    if res_str.startswith('NTS'): return 'NTS'
    
    # 2. 蛋白部分：提取数字判定所在域
    match = re.search(r'(\d+)', res_str)
    if not match: return 'Other'
    
    resid = int(match.group(1))
    for dom, segments in domain_defs.items():
        for start, end in segments:
            if start <= resid <= end: return dom
            
    return 'Other'

# ==========================================
# 2. 加载残基数据并聚合 (去除了所有冗余对齐代码)
# ==========================================
master_dict = {}
for i, rs in enumerate(states):
    file_path = f"{rs}/{FILE_NAME}"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            # 数据已经是完美的，直接读取！
            pos = str(row['Pos_Residue'])
            tgt = str(row['Target_Residue'])
            
            pair = tuple(sorted([pos, tgt]))
            if pair not in master_dict:
                master_dict[pair] = [0.0] * 7
            master_dict[pair][i] = max(master_dict[pair][i], row['Occupancy'])

if not master_dict:
    print(f"错误: 未能读取到任何 {FILE_NAME} 数据，请检查当前路径。")
    exit()

df_master = pd.DataFrame.from_dict(master_dict, orient='index', columns=display_states)
df_bin = (df_master >= CUTOFF).astype(int)
df_bin = df_bin[df_bin.sum(axis=1) > 0] 
df_bin = df_bin[df_bin.sum(axis=1) < 7] 

domain_networks = {state: nx.Graph() for state in display_states}
G_global_domain = nx.Graph()

for pair, row in df_bin.iterrows():
    dom1 = get_domain(pair[0])
    dom2 = get_domain(pair[1])
    
    if dom1 == dom2 or dom1 == 'Other' or dom2 == 'Other':
        continue
        
    for state in display_states:
        if row[state] == 1:
            if not G_global_domain.has_edge(dom1, dom2):
                G_global_domain.add_edge(dom1, dom2, weight=0)
                
            if not domain_networks[state].has_edge(dom1, dom2):
                domain_networks[state].add_edge(dom1, dom2, weight=1)
            else:
                domain_networks[state][dom1][dom2]['weight'] += 1

# ==========================================
# 3. 计算全局坐标 (Circular Layout)
# ==========================================
print(f"正在构建跨结构域变构网络布局...")
pos_global = nx.circular_layout(G_global_domain)

# ==========================================
# 4. 绘制 2x4 双行排版宏观变构演化图
# ==========================================
print("正在生成 2x4 结构域层级图表...")
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor('white')
axes = axes.flatten()

for i, state in enumerate(display_states):
    ax = axes[i]
    G_current = domain_networks[state]
    
    if len(G_current.edges()) == 0:
        ax.set_title(f"{state}", fontsize=18, fontweight='bold', pad=10)
        ax.axis('off')
        continue
    
    active_nodes = list(G_current.nodes())
    current_pos = {node: pos_global[node] for node in active_nodes}
    node_colors = [domain_colors.get(node, '#808080') for node in active_nodes]
    
    edge_weights = [G_current[u][v]['weight'] for u, v in G_current.edges()]
    scaled_widths = [w * 1.5 for w in edge_weights] 
    
    nx.draw_networkx_edges(G_current, current_pos, ax=ax, alpha=0.6, 
                           width=scaled_widths, edge_color='darkgrey', 
                           connectionstyle="arc3,rad=0.15")
    
    nx.draw_networkx_nodes(G_current, current_pos, ax=ax, node_color=node_colors, 
                           node_size=800, linewidths=0)
    
    labels = {node: node for node in active_nodes}
    nx.draw_networkx_labels(G_current, current_pos, labels=labels, ax=ax, 
                            font_size=9, font_family="sans-serif", font_weight='bold')
    
    ax.set_title(f"{state}", fontsize=18, fontweight='bold', pad=10)
    ax.axis('off') 
    ax.margins(0.10)

# ==========================================
# 5. 第 8 个子图位置绘制图例 Legend
# ==========================================
ax_legend = axes[7]
ax_legend.axis('off')

all_active_domains = list(G_global_domain.nodes())
legend_patches = [mpatches.Patch(color=domain_colors[dom], label=dom) 
                  for dom in domain_order if dom in all_active_domains]

ax_legend.legend(handles=legend_patches, loc='center', ncol=2, 
                 fontsize=12, frameon=False, title="Domain Legend", title_fontsize=15)

plt.suptitle(f"Macroscopic Inter-domain Communication Network ({TITLE_NAME})", 
             fontsize=24, fontweight='bold', y=1.02)
plt.tight_layout()

output_img = f"{PREFIX}_Macroscopic_Domain_Network_Final.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
print(f"✓ 已完美适配全新底层数据，图片生成完毕: {output_img}")
