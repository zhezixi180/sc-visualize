import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
from config import *

# 配置绘图参数 (类似于 Seurat 的默认风格)
sc.settings.verbosity = 3             # 打印详细日志
sc.settings.set_figure_params(dpi=150, facecolor='white', frameon=False, vector_friendly=True) # 高质量绘图
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "figure.titlesize": 13,
})

def run_qc_pipeline():
    print("=== Step 2: Quality Control ===")
    print(f"Run start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 1. 读取数据
    print("Loading raw merged dataset...")
    input_file = SPINAL_DIR / "spinal_all_raw.h5ad"
    if not input_file.exists():
        print(f"Error: {input_file} not found. Please run make_spinal_dataset.py first.")
        return

    adata = sc.read_h5ad(input_file)
    print(f"Initial shape: {adata.shape}")
    
    # 2. 计算 QC 指标 (复刻 Seurat: PercentageFeatureSet)
    # 标记线粒体基因 (以 "mt-" 或 "Mt-" 开头)
    # 小鼠通常是 "mt-" (小写) 或 "Mt-" (大写开头)，保险起见都匹配
    adata.var['mt'] = adata.var_names.str.startswith('mt-') | adata.var_names.str.startswith('Mt-') 
    
    # 计算 QC metrics: n_genes_by_counts, total_counts, pct_counts_mt
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

    # 3. 绘制 QC 小提琴图 (复刻 Seurat: VlnPlot)
    print("Plotting QC violin plots...")
    
    # 确保 figures 目录存在，并设为 scanpy 默认输出目录
    fig_dir = SPINAL_DIR.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sc.settings.figdir = fig_dir

    def _upper_lim(series, q=0.99, min_val=1.0):
        """用于绘图的上限，裁掉极端值，避免白图/尖峰。"""
        val = np.nanpercentile(series, q * 100)
        return max(val, min_val)

    def _set_violin_ylim(axes, ylims):
        for ax, ylim in zip(axes, ylims):
            ax.set_ylim(0, ylim)

    def _plot_violin_main(adata, metrics, ylims, out_path):
        title_map = {
            "n_genes_by_counts": "Genes per cell",
            "total_counts": "Total counts per cell",
            "pct_counts_mt": "Mitochondrial %",
        }
        sc.pl.violin(
            adata,
            metrics,
            stripplot=False,
            multi_panel=True,
            show=False,
        )
        for ax, key, ylim in zip(plt.gcf().axes, metrics, ylims):
            ax.set_title(title_map.get(key, key))
            ax.set_xlabel("")
            ax.set_ylabel(key)
            ax.set_ylim(0, ylim)
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()

    # 保存 QC 前的图
    # stripplot=False: 不画黑点，只画小提琴形状，避免大量数据点导致的内存溢出和图片全黑
    before_ylims = [
        _upper_lim(adata.obs['n_genes_by_counts'], q=0.95),
        _upper_lim(adata.obs['total_counts'], q=0.95),
        _upper_lim(adata.obs['pct_counts_mt'], q=0.95),
    ]
    _plot_violin_main(
        adata,
        ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
        before_ylims,
        fig_dir / "violin_before_filtering.png",
    )
    
    # 绘制散点图 (复刻 Seurat: FeatureScatter)
    # 看基因数 vs 线粒体比例
    # 使用 matplotlib 手动设置 limit，排除前 1% 的极端值，避免图被个别离群点拉伸成白纸
    x_limit = _upper_lim(adata.obs['total_counts'], q=0.99)
    y_limit_mt = _upper_lim(adata.obs['pct_counts_mt'], q=0.99)
    y_limit_genes = _upper_lim(adata.obs['n_genes_by_counts'], q=0.99)

    # 1. Counts vs MT
    mask_mt = (adata.obs['total_counts'] <= x_limit) & (adata.obs['pct_counts_mt'] <= y_limit_mt)
    adata_mt = adata[mask_mt].copy()
    sc.pl.scatter(adata_mt, x='total_counts', y='pct_counts_mt', show=False)
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit_mt)
    plt.savefig(fig_dir / "scatter_counts_vs_mt.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Counts vs Genes
    mask_genes = (adata.obs['total_counts'] <= x_limit) & (adata.obs['n_genes_by_counts'] <= y_limit_genes)
    adata_genes = adata[mask_genes].copy()
    sc.pl.scatter(adata_genes, x='total_counts', y='n_genes_by_counts', show=False)
    plt.xlim(0, x_limit)
    plt.ylim(0, y_limit_genes)
    plt.savefig(fig_dir / "scatter_counts_vs_genes.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 执行过滤 (复刻 Seurat: subset)
    print("Applying QC filters...")
    print(f"Before filter: {adata.n_obs} cells")
    
    # 师兄的标准: min.features = 200 (n_genes_by_counts >= 200)
    # 师兄的标准: min.cells = 3 (在 make_spinal_dataset.py 之前可能没做，通常在这里补上 gene fliter)
    # 师兄的标准: percent.mt < 5 (但对于 spinal cord 有时候可以放宽到 10-20，这里先设 20 作为宽容阈值，或者严格按 Seurat 设 5)
    # 建议先看图再决定，这里暂定: min_genes=200, max_mt=20 (比师兄宽容一点，防止杀太狠)
    
    # 过滤基因：至少在 3 个细胞中表达
    sc.pp.filter_genes(adata, min_cells=3)
    
    # 过滤细胞：基因数 < 200 或者 > 6000 (排除双细胞)
    # 过滤细胞：线粒体 > 20% (如果师兄是 5%，可以改成 5)
    sc.pp.filter_cells(adata, min_genes=200)
    # sc.pp.filter_cells(adata, max_genes=6000) # 可选：去除 Doublets
    
    # 过滤线粒体
    adata = adata[adata.obs.pct_counts_mt < 5, :] # 这里先设 20，如果师兄那是严格的 5，请改成 5
    
    print(f"After filter: {adata.n_obs} cells, {adata.n_vars} genes")

    # 保存 QC 后的图
    after_ylims = [
        _upper_lim(adata.obs['n_genes_by_counts'], q=0.95),
        _upper_lim(adata.obs['total_counts'], q=0.95),
        _upper_lim(adata.obs['pct_counts_mt'], q=0.95),
    ]
    _plot_violin_main(
        adata,
        ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
        after_ylims,
        fig_dir / "violin_after_filtering.png",
    )

    # 5. 保存结果
    output_file = SPINAL_DIR / "spinal_all_qc.h5ad"
    print(f"Saving QC metrics to {output_file}...")
    adata.write(output_file)
    print("QC Pipeline Done.")

if __name__ == "__main__":
    run_qc_pipeline()
