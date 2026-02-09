# import torch
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse
from pathlib import Path
import sys

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
try:
    from config import *
except ImportError:
    # 允许独立运行时的路径回退逻辑
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_DIR / "data"
    SIGNAL_DB_DIR = PROJECT_DIR / "signal_db" 
    TABLES_DIR = PROJECT_DIR / "output" / "tables"
    MATRIX_DIR = PROJECT_DIR / "output" / "matrices"

# =================
# 针对新数据的路径覆盖
# =================
SIGNAL_DB_DIR = PROJECT_DIR / "signal_db"
DRG_FILE = SIGNAL_DB_DIR / "DRG_neuron_SNI_all.h5ad"
SPINAL_FILE = DATA_DIR / "SC_Neuron_merge_1.h5ad"

# 输出目录
TABLES_DIR = PROJECT_DIR / "output" / "tables"
MATRIX_DIR = PROJECT_DIR / "output" / "matrices"

def load_drg_data(path):
    """
    加载 DRG 数据并识别细胞类型列。
    模拟 R 代码中对 filtered Seurat 对象的处理。
    """
    print(f"正在从 {path} 加载 DRG 数据...")
    try:
        adata = sc.read_h5ad(path)
        obs_cols = adata.obs.columns
        target_col = None
        
        # DRG 细胞类型列的优先级列表
        priorities = ['celltype_2', 'celltype_sample', 'celltype', 'ident']
        for col in priorities:
            if col in obs_cols:
                target_col = col
                break
        
        if target_col:
            print(f"  使用 '{target_col}' 作为 DRG 细胞类型分组。")
            adata.obs['cell_type_mapped'] = adata.obs[target_col]
        else:
            raise ValueError(f"在 DRG 数据中找不到细胞类型列。现有列: {list(obs_cols)}")
            
        return adata
    except Exception as e:
        print(f"加载 DRG 数据出错: {e}")
        return None

def load_spinal_data(path):
    """
    加载脊髓数据，映射细胞类型和层级信息。
    对应 R 代码中 load(file = 'SC_Neuron_merge.Rdata') 的部分。
    """
    print(f"正在从 {path} 加载脊髓数据...")
    try:
        adata = sc.read_h5ad(path)
        obs_cols = adata.obs.columns
        
        # 1. 识别细胞类型
        cell_col = None
        for col in ['cluster', 'Cluster']:
            if col in obs_cols:
                cell_col = col
                break
        # 兜底方案
        if not cell_col: cell_col = obs_cols[0]
        
        print(f"  使用 '{cell_col}' 作为脊髓细胞类型。")
        adata.obs['cell_type_mapped'] = adata.obs[cell_col]

        # 2. 识别层级 (Layer/Lamina)
        layer_col = None
        for col in ['lamina', 'Lamina', 'layer', 'Layer', 'layers']:
            if col in obs_cols:
                layer_col = col
                break
        
        # 模糊搜索
        if not layer_col:
            for col in obs_cols:
                if 'layer' in col.lower() or 'lamina' in col.lower():
                    layer_col = col
                    break
        
        if layer_col:
            print(f"  使用 '{layer_col}' 作为脊髓层级信息。")
            # 确保转换为字符串并处理缺失值
            adata.obs['layer_mapped'] = adata.obs[layer_col].astype(str).replace({'nan': 'Unknown', 'None': 'Unknown'})
        else:
            print("  警告: 未找到层级列！将统一设为 Unknown。")
            adata.obs['layer_mapped'] = "Unknown"

        return adata
    except Exception as e:
        print(f"加载脊髓数据出错: {e}")
        return None

def get_mean_expression(adata, genes, group_col):
    """
    计算分组平均表达量。
    返回 DataFrame: 索引=分组, 列=基因
    对应 R 中的 mean_expression 和 rowMeans 逻辑。
    """
    # 过滤数据集中存在的基因
    genes_exist = [g for g in genes if g in adata.var_names or (adata.raw and g in adata.raw.var_names)]
    if not genes_exist:
        return pd.DataFrame()
    
    results = {}
    groups = adata.obs[group_col].unique()
    
    # 1. 在主矩阵 X 中查找
    genes_in_X = [g for g in genes_exist if g in adata.var_names]
    if genes_in_X:
        sub = adata[:, genes_in_X]
        for g in groups:
            mask = adata.obs[group_col] == g
            if not mask.any(): continue
            
            chunk = sub[mask].X
            if scipy.sparse.issparse(chunk):
                # 使用 .A1 或 ravel 展平以便存入 list
                vals = np.array(chunk.mean(axis=0)).flatten()
            else:
                vals = np.mean(chunk, axis=0)
            
            if g not in results: results[g] = {}
            for i, gene in enumerate(genes_in_X):
                results[g][gene] = vals[i]
                
    # 2. 在 Raw 矩阵中查找 (补充)
    genes_in_raw = [g for g in genes_exist if g not in adata.var_names and adata.raw and g in adata.raw.var_names]
    if genes_in_raw:
        raw_X = adata.raw.X
        indices = [adata.raw.var_names.get_loc(g) for g in genes_in_raw]
        
        for g in groups:
            mask = adata.obs[group_col] == g
            if not mask.any(): continue
            
            cells = np.where(mask)[0]
            chunk = raw_X[cells][:, indices]
            
            if scipy.sparse.issparse(chunk):
                vals = np.array(chunk.mean(axis=0)).flatten()
            else:
                vals = np.mean(chunk, axis=0)
                
            if g not in results: results[g] = {}
            for i, gene in enumerate(genes_in_raw):
                results[g][gene] = vals[i]
                
    df = pd.DataFrame.from_dict(results, orient='index')
    return df.fillna(0)

def main():
    print("=== 步骤 5: DRG -> 脊髓 互作分析 (严格对齐 R 逻辑) ===")
    
    # 1. 构建信号数据库
    # 此处硬编码的配对列表是为了严格复刻 sc-drg-connetcion_3.R 中的自定义数据库构建过程
    pairs_full = [
        # 痒觉 (Itch)
        ("Nppb",     "Npr1",      "peptide",     "itch",    "NPPB"),
        ("Grp",      "Grpr",      "peptide",     "itch",    "GRP"),
        # 痛觉 (Pain)
        ("Tac1",     "Tacr1",     "peptide",     "pain",    "SubstanceP"),
        ("Calca",    "Calcrl",    "peptide",     "pain",    "CGRP"),
        ("Calca",    "Ramp1",     "peptide",     "pain",    "CGRP"),
        ("Adcyap1",  "Adcyap1r1", "peptide",     "pain",    "PACAP"),
        # 调控/共享 (Modulatory/Shared)
        ("Penk",     "Oprd1",     "modulatory",  "shared",  "Opioid"),
        ("Penk",     "Oprm1",     "modulatory",  "shared",  "Opioid"),
        ("Pdyn",     "Oprk1",     "modulatory",  "shared",  "Opioid"),
        ("Sst",      "Sstr1",     "modulatory",  "shared",  "SST"),
        ("Sst",      "Sstr2",     "modulatory",  "shared",  "SST"),
        ("Gal",      "Galr1",     "modulatory",  "shared",  "Galanin"),
        ("Gal",      "Galr2",     "modulatory",  "shared",  "Galanin")
    ]
    db = pd.DataFrame(pairs_full, columns=["ligand", "receptor", "signal_level", "modality", "pathway"])
    
    # 2. 加载数据
    adata_drg = load_drg_data(DRG_FILE)
    adata_spinal = load_spinal_data(SPINAL_FILE)
    if not adata_drg or not adata_spinal: return

    # 3. 计算均值矩阵
    print("计算 DRG 配体均值...")
    drg_means = get_mean_expression(adata_drg, db['ligand'].unique(), 'cell_type_mapped')
    
    print("计算脊髓受体均值...")
    spinal_means = get_mean_expression(adata_spinal, db['receptor'].unique(), 'cell_type_mapped')
    
    drg_types = sorted(drg_means.index.tolist())
    spinal_types = sorted(spinal_means.index.tolist())
    
    # 4. 计算每对 LR 的归一化矩阵
    # 对应 R 代码逻辑: score <- outer(S, R); score <- score / max(score)
    print("计算互作得分并执行 R 风格归一化 (Score / Max)...")
    
    lr_matrices = [] # 存储字典: {modality, matrix, pathway, ligand, receptor}
    
    contributions = [] # 用于从属关系追踪
    
    for _, row in db.iterrows():
        lig = row['ligand']
        rec = row['receptor']
        mod = row['modality'] # pain/itch/shared
        path = row['pathway']
        
        if lig not in drg_means.columns or rec not in spinal_means.columns:
            continue
            
        # 重新索引以确保顺序一致
        vec_s = drg_means.loc[drg_types, lig].values # (N_drg,)
        vec_r = spinal_means.loc[spinal_types, rec].values # (N_spinal,)
        
        # 外积计算: S * R
        raw_mat = np.outer(vec_s, vec_r)
        
        # === R 逻辑: 归一化 ===
        # 这里对每个矩阵单独做 max 归一化，使得不同表达丰度的通路权重可比
        max_val = np.max(raw_mat)
        if max_val > 0:
            norm_mat = raw_mat / max_val
        else:
            norm_mat = raw_mat
            
        lr_matrices.append({
            "matrix": norm_mat,
            "modality": mod,
            "pathway": path,
            "ligand": lig,
            "receptor": rec
        })
        
        # 收集贡献度 (使用归一化后的分数)
        df_score = pd.DataFrame(norm_mat, index=drg_types, columns=spinal_types)
        mask = df_score > 0.001 # 仅保留非微小值
        if mask.any().any():
            rows, cols = np.where(mask)
            for r_idx, c_idx in zip(rows, cols):
                c_val = norm_mat[r_idx, c_idx]
                if c_val > 0.05: # 输出阈值控制
                    contributions.append({
                        'drg': drg_types[r_idx],
                        'sc': spinal_types[c_idx],
                        'ligand': lig,
                        'receptor': rec,
                        'pathway': path,
                        'modality': mod,
                        'value': c_val
                    })

    # 5. 按模态 (Modality) 聚合
    # R 逻辑: Reduce('+', mats) / length(mats) -> 即求平均
    print("按模态聚合矩阵 (取平均值)...")
    
    # 将矩阵按 mod 分组
    mod_groups = {}
    for item in lr_matrices:
        m = item['modality']
        if m not in mod_groups: mod_groups[m] = []
        mod_groups[m].append(item['matrix'])
        
    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    
    for mod, mat_list in mod_groups.items():
        if not mat_list: continue
        
        # 求和
        sum_mat = np.zeros((len(drg_types), len(spinal_types)))
        for m in mat_list:
            sum_mat += m
            
        # 求平均
        mean_mat = sum_mat / len(mat_list)
        
        df_out = pd.DataFrame(mean_mat, index=drg_types, columns=spinal_types)
        
        # 为了与可视化代码兼容，将 'shared' 重命名为 'modulatory'
        out_name = mod
        if out_name == 'shared': out_name = 'modulatory'
        
        fname = MATRIX_DIR / f"{out_name}_mat.csv"
        df_out.to_csv(fname)
        print(f"  已保存 {out_name} 矩阵 (平均了 {len(mat_list)} 对) 到 {fname}")

    # 6. 计算层级分布
    # R 逻辑: freq = n / sum(n)
    print("计算脊髓层级分布...")
    layer_counts = adata_spinal.obs.groupby(['cell_type_mapped', 'layer_mapped']).size().reset_index(name='count')
    layer_counts['total'] = layer_counts.groupby('cell_type_mapped')['count'].transform('sum')
    layer_counts['freq'] = layer_counts['count'] / layer_counts['total']
    
    layer_dist_out = layer_counts[['cell_type_mapped', 'layer_mapped', 'freq']].rename(columns={
        'cell_type_mapped': 'cell_type',
        'layer_mapped': 'layer'
    })
    
    ld_file = MATRIX_DIR / "layer_dist.csv"
    layer_dist_out.to_csv(ld_file, index=False)
    print(f"  已保存层级分布到 {ld_file}")
    
    # 7. 保存明细贡献表
    if contributions:
        df_contrib = pd.DataFrame(contributions)
        # 为 R 代码 sc-drg-connetcion_3.R 中的 'Sankey_Input_Details' 对应物
        ct_file = MATRIX_DIR / "lr_edge_contributions.csv"
        df_contrib.to_csv(ct_file, index=False)
        print(f"  已保存详细贡献表到 {ct_file}")

    print("=== 分析完成 (已对齐 R 代码逻辑) ===")

if __name__ == "__main__":
    main()
