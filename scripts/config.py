import os
from pathlib import Path

# =======================
# 路径配置
# =======================
# 获取当前脚本所在目录的父级目录 (即 SC_project 项目根目录)
PROJECT_DIR = Path(__file__).resolve().parent.parent

# 数据目录
DATA_DIR = PROJECT_DIR / "data"
# 数据直接位于 data 目录下
SPINAL_DIR = DATA_DIR 
# DRG_DIR = DATA_DIR / "drg"  # 您的数据中没有此目录

# 输出目录
OUTPUT_DIR = PROJECT_DIR / "output"
MATRIX_DIR = OUTPUT_DIR / "matrices"
QC_DIR = OUTPUT_DIR / "qc"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = PROJECT_DIR / "figures"  # 新增图片保存路径

# 映射文件与数据库
MAPPING_DIR = PROJECT_DIR / "mapping"
SIGNAL_DB_PATH = PROJECT_DIR / "signal_db" / "signal_db.csv"

# =======================
# 分析参数配置 (复刻 Seurat 流程)
# =======================
# 1. 过滤 (QC)
MIN_GENES = 200      # Seurat: min.features = 200
MIN_CELLS = 3        # Seurat: min.cells = 3
MAX_MITO_PCT = 20.0  # 线粒体比例阈值 (常用经验值，可视情况调整)

# 2. 预处理
N_TOP_GENES = 2000   # Seurat: nfeatures = 2000
PCA_N_COMPONENTS = 30 # Seurat: npcs = 30

# 3. 聚类
LEIDEN_RESOLUTION = 1.2 # Seurat: resolution = 1.2

# 4. 绘图
FIGURE_SIZE = (6, 6)

# 确保输出目录存在
for p in [MATRIX_DIR, QC_DIR, TABLES_DIR]:
    p.mkdir(parents=True, exist_ok=True)
