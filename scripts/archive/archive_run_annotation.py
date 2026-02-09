import torch
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
from config import *

def plot_basic_umap(adata):
    """绘制基础 UMAP，按 Cluster 和 Dataset 着色"""
    print("Plotting UMAP...")
    
    # 确保 figures 目录存在
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 全局绘图设置
    sc.settings.figdir = FIGURES_DIR
    sc.set_figure_params(scanpy=True, fontsize=12, dpi=300, facecolor='white', figsize=FIGURE_SIZE)

    # 1. Plot Clustering (Leiden)
    sc.pl.umap(
        adata, 
        color=['leiden'], 
        legend_loc='on data', 
        title='Clustering (Leiden)',
        show=False,
        save='_clusters.png'
    )
    
    # 2. Plot Dataset (Batch effect check)
    sc.pl.umap(
        adata, 
        color=['dataset'], 
        title='Source Datasets',
        show=False,
        save='_datasets.png'
    )
    print(f"  Saved UMAPs to {FIGURES_DIR}")

def plot_markers(adata, n_genes=5):
    """绘制 Marker 基因的气泡图和热图"""
    print("Plotting Marker Genes...")
    
    # 检查 rank_genes_groups 是否存在
    if 'rank_genes_groups' not in adata.uns:
        print("  Running rank_genes_groups...")
        sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')

    # 获取每个 Cluster 的 Top N 基因
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # 提取基因列表用于绘图
    top_genes = pd.DataFrame(result['names']).head(n_genes)
    marker_genes_dict = {group: top_genes[group].tolist() for group in groups}
    
    # 1. DotPlot (最常用，直观)
    print("  Generating DotPlot...")
    sc.pl.rank_genes_groups_dotplot(
        adata, 
        n_genes=n_genes, 
        values_to_plot="logfoldchanges", 
        cmap='bwr',
        vmin=-4, 
        vmax=4, 
        min_logfoldchange=1.0, 
        colorbar_title='log fold change',
        show=False,
        save='_markers.png' # scanpy会自动添加前缀 dotplot_
    )

    # 2. MatrixPlot (适合基因很多时)
    print("  Generating MatrixPlot...")
    sc.pl.matrixplot(
        adata, 
        marker_genes_dict, 
        groupby='leiden', 
        cmap='viridis', 
        standard_scale='var',
        show=False,
        save='_markers_matrix.png'
    )

def export_marker_table(adata, top_n=50):
    """导出详细的 Marker 表到 CSV"""
    print(f"Exporting Top {top_n} markers to CSV...")
    
    sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    
    # 构建 DataFrame
    df_list = []
    for group in groups:
        df = pd.DataFrame({
            'cluster': group,
            'gene': [result['names'][i][group] for i in range(top_n)],
            'score': [result['scores'][i][group] for i in range(top_n)],
            'pvals_adj': [result['pvals_adj'][i][group] for i in range(top_n)],
            'logfoldchanges': [result['logfoldchanges'][i][group] for i in range(top_n)]
        })
        df_list.append(df)
    
    markers_df = pd.concat(df_list)
    
    out_file = TABLES_DIR / "cluster_markers.csv"
    markers_df.to_csv(out_file, index=False)
    print(f"  Saved marker table to {out_file}")

def main():
    print("=== 4. Visualization & Annotation Helper ===")
    
    # 1. Load Data
    data_path = DATA_DIR / "spinal_all_processed.h5ad"
    print(f"Loading data from {data_path}...")
    if not data_path.exists():
        print("Error: Processed data not found. Run run_clustering.py first.")
        return
        
    adata = sc.read_h5ad(data_path)
    print(f"Data loaded: {adata.shape}")

    # 2. Plots
    plot_basic_umap(adata)
    plot_markers(adata, n_genes=5)
    
    # 3. Export Table
    export_marker_table(adata)
    
    print("\n[Done]")
    print(f"1. Check visualizations in: {FIGURES_DIR}")
    print(f"2. Check marker table in: {TABLES_DIR}")
    print("3. Based on these, determine the Cell Type for each Cluster.")
    print("   Then update the 'annotation_dict' in step5_annotate.py (to be created).")

if __name__ == "__main__":
    main()
