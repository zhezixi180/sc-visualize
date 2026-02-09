
import torch
import scanpy as sc
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
from config import *

def main():
    print("=== Step 4: Applying Cell Type Annotations ===")
    
    # 1. 加载数据
    data_path = DATA_DIR / "spinal_all_processed.h5ad"
    if not data_path.exists():
        print("Error: Input data not found.")
        return
    
    adata = sc.read_h5ad(data_path)
    print(f"Loaded data: {adata.shape}")

    # 2. 定义注释字典 (根据刚才的分析)
    # 格式: { 'Cluster_ID': 'Cell Name' }
    annotation_dict = {
        '0': 'Endothelial',         # Cldn5, Flt1
        '1': 'Neuron (Subtype A)',  # Rbfox3
        '2': 'Neuron (Subtype B)',  # Celf2, Ebf1
        '3': 'Neuron (Subtype C)',  # Adcy2
        '4': 'Oligodendrocytes',    # Mbp, Plp1 (Strong)
        '5': 'Neuron (Subtype D)',  # Tns1
        '6': 'Neuron (Subtype E)',  # Nos1ap
        '7': 'Low Quality',         # High Mito
        '8': 'Inhibitory Neuron',   # Gad2
        '9': 'Astrocytes',          # Slc1a2, Slc4a4
        '10': 'Oligodendrocytes',   # Mbp (Weaker)
        '11': 'Neuron (Subtype F)', # Pou6f2
        '12': 'Neuron (Subtype G)', # Kitl
        '13': 'OPC',                # Vcan, Sox6
        '14': 'Ependymal',          # Dnah12
        '15': 'Excitatory Neuron',  # Slc17a6
        '16': 'Astrocytes',         # Gfap
        '17': 'Endothelial',        # Flt1
        '18': 'Neuron (Subtype H)', # Ptprd
        '19': 'Ependymal',          # Foxj1/Cilia
        '20': 'Low Quality',        # High Mito
        '21': 'Immune/Vascular'     # Fli1
    }

    # 3. 应用注释
    # 将 leiden (0,1,2...) 映射为 cell_type
    print("Applying annotations...")
    adata.obs['cell_type'] = adata.obs['leiden'].map(annotation_dict)
    
    # 将 cell_type 转为 categorical 类型以节省内存
    adata.obs['cell_type'] = adata.obs['cell_type'].astype('category')

    # 4. 绘图验证
    print("Plotting annotated UMAP...")
    sc.settings.figdir = FIGURES_DIR
    sc.set_figure_params(dpi=300, facecolor='white', figsize=(8, 8))
    
    sc.pl.umap(
        adata, 
        color=['cell_type'], 
        legend_loc='on data', 
        legend_fontoutline=2,
        title='Annotated Cell Types',
        show=False,
        save='_annotated.png'
    )

    # 5. 保存结果
    # 我们通常把这作为最终文件保存
    output_path = DATA_DIR / "spinal_all_annotated.h5ad"
    print(f"Saving fully annotated data to {output_path}...")
    adata.write(output_path)
    
    print("\n[Done] Pipeline Complete.")
    print(f"Annotated UMAP saved to: {FIGURES_DIR / 'umap_annotated.png'}")

if __name__ == "__main__":
    main()
