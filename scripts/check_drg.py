import scanpy as sc
import sys

try:
    adata = sc.read_h5ad(r'd:\pythondaima\单细胞分析\project_2\SC_project\signal_db\DRG_neuron_SNI_all.h5ad')
    print("Columns:", list(adata.obs.columns))
    if 'celltype' in adata.obs:
        print("celltype sample:", adata.obs['celltype'].unique().tolist()[:10])
    if 'celltype_2' in adata.obs:
        print("celltype_2 sample:", adata.obs['celltype_2'].unique().tolist()[:10])
    if 'celltype_sample' in adata.obs:
        print("celltype_sample sample:", adata.obs['celltype_sample'].unique().tolist()[:10])
except Exception as e:
    print(e)
