import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import os
import sys
import scipy.io
import gzip
import shutil
import gc
from scipy import sparse

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
from config import *

def clean_adata(adata, dataset_name):
    """通用清理函数: 确保基因名唯一，添加 dataset 标签"""
    adata.var_names_make_unique()
    adata.obs['dataset'] = dataset_name
    return adata

def load_GSE167597():
    """
    [2] GSE167597 (Alkaslasi)
    文件:
    - GSE167597_allMN_countsmatrix.csv.gz (矩阵)
    - GSE167597_all_AlkaslasiPiccus_annotations.csv.gz (注释)
    """
    print("\n[2] Loading GSE167597 (Alkaslasi)...")
    data_dir = SPINAL_DIR / "GSE167597"
    counts_file = data_dir / "GSE167597_allMN_countsmatrix.csv.gz"
    meta_file = data_dir / "GSE167597_all_AlkaslasiPiccus_annotations.csv.gz"
    
    if not counts_file.exists():
        print(f"  Skipping: {counts_file} not found")
        return None
        
    print("  Reading counts (CSV)...")
    # 假设也是 Gene x Cell，如果报错或者维度反了需要检查
    try:
        df = pd.read_csv(counts_file, index_col=0)
        df.fillna(0, inplace=True) # 填充可能存在的空值
        # 检查是否需要转置: 通常基因数(2w-3w) < 细胞数，或者看index是不是基因名
        # 简单判断: 如果 columns 有 'Actb' 等基因名，则不需要转置。这里假设是 Dense Matrix Gene x Cell
        adata = sc.AnnData(df.T)
        adata.X = sparse.csr_matrix(adata.X)
        del df
        gc.collect()
    except Exception as e:
        print(f"  Error reading counts: {e}")
        return None
        
    # 读取注释
    if meta_file.exists():
        print("  Reading metadata...")
        try:
            meta = pd.read_csv(meta_file, index_col=0)
            # 尝试匹配索引
            common = adata.obs_names.intersection(meta.index)
            if len(common) > 0:
                adata = adata[common].copy()
                adata.obs = adata.obs.join(meta)
                print(f"  Matched {len(common)} cells with metadata.")
            else:
                 print("  Warning: No matching cells found in metadata.")
        except Exception as e:
            print(f"  Error reading metadata: {e}")

    return clean_adata(adata, "GSE167597")

def load_GSE161621():
    """
    [5] GSE161621 (Jacob)
    文件: 多个 .h5
    """
    print("\n[5] Loading GSE161621 (Jacob)...")
    data_dir = SPINAL_DIR / "GSE161621"
    h5_files = list(data_dir.glob("*.h5"))
    
    if not h5_files:
        print("  Skipping: No .h5 files found")
        return None
        
    adatas = []
    print(f"  Found {len(h5_files)} H5 files. Loading...")
    for f in h5_files:
        try:
            print(f"    Loading {f.name}...")
            # 必须使用 var_names='gene_symbols' 防止使用 Ensembl ID 导致无法合并
            # 之前报错可能是因为 index 问题，这里尝试加上 explicit var_names
            ad_tmp = sc.read_10x_h5(f)
            ad_tmp.var_names_make_unique()
            
            # 过滤空液滴 (Empty Droplets)
            # 原始 H5 文件可能包含背景噪音（数百万个几乎为空的 barcodes），必须先过滤，否则内存溢出
            n_raw = ad_tmp.n_obs
            sc.pp.filter_cells(ad_tmp, min_counts=100)
            print(f"    Filtered empty droplets: {n_raw} -> {ad_tmp.n_obs} cells (min_counts=100)")
            
            # 记录来源于样本名
            # 使用 split('_', 1)[1] 来保留 GSM ID 之后的所有部分，确保样本名唯一
            # 例如 GSM4911290_mixed4_5_31_a -> mixed4_5_31_a
            sample_name = f.stem.split('_', 1)[1] 
            ad_tmp.obs['sample'] = sample_name
            adatas.append(ad_tmp)
        except Exception as e:
            print(f"    Error loading {f.name}: {e}")
            
    if not adatas:
        return None
        
    print("  Concatenating GSE161621 samples...")
    # 合并该数据集内部的多个样本
    adata_full = ad.concat(adatas, index_unique="_", join="outer")
    return clean_adata(adata_full, "GSE161621")

def main():
    # 使用临时目录存储中间文件
    temp_dir = SPINAL_DIR / "processed_temp"
    temp_dir.mkdir(exist_ok=True)
    
    loaders = [
        ("GSE167597", load_GSE167597),
        ("GSE161621", load_GSE161621)
    ]
    
    processed_files = []
    
    # === 阶段 1: 逐个加载并转换 ===
    print("=== Phase 1: Sequential Loading & Converting ===")
    for name, load_func in loaders:
        out_path = temp_dir / f"{name}.h5ad"
        processed_files.append(out_path)
        
        # 如果已经存在，可以选择跳过（这里为了保险还是重新跑），或者直接跳过
        if out_path.exists():
            print(f"  Skipping {name}: Found existing {out_path} (delete it to re-run)")
            continue
            
        print(f"  Processing {name}...")
        try:
            adata = load_func()
            if adata is None:
                print(f"    Warning: {name} loading returned None.")
                continue
            
            # 再次确认是稀疏矩阵（loader里已经转了，这里双保险）
            if not sparse.issparse(adata.X):
                adata.X = sparse.csr_matrix(adata.X)
            
            print(f"    Saving intermediate to: {out_path.name}")
            adata.write(out_path)
            
            # 释放内存
            del adata
            gc.collect()
            print(f"    Finished {name}. Memory cleared.")
            
        except Exception as e:
            print(f"    CRITICAL ERROR loading {name}: {e}")

    # === 阶段 2: 合并所有数据集 ===
    print("\n=== Phase 2: Merging All Datasets ===")
    adatas = []
    
    for p in processed_files:
        if p.exists():
            print(f"  Loading cached: {p.name}")
            # backed='r' 可以不读入内存模式，但 concat 需要读入内存。
            # 这里我们读取稀疏矩阵的 h5ad，内存消耗远小于原始 CSV
            ad_p = sc.read_h5ad(p)
            adatas.append(ad_p)
    
    if not adatas:
        print("No datasets successfully processed. Exiting.")
        return

    print(f"  Concatenating {len(adatas)} datasets...")
    # Outer join 保留所有基因，index_unique 添加后缀防止细胞名冲突
    spinal_all = ad.concat(adatas, join="outer", index_unique="-")
    
    print(f"Final Merged Shape: {spinal_all.shape}")
    
    # Save
    out_file = SPINAL_DIR / "spinal_all_raw.h5ad"
    print(f"Saving to {out_file}...")
    spinal_all.write(out_file)
    print("Done.")

if __name__ == "__main__":
    print("=== Step 1: Make Spinal Dataset ===")
    main()
