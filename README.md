# Single Cell Visualization Project (SC-Visualize)

这是一个基于 Streamlit 的单细胞数据交互分析与可视化平台。主要用于展示脊髓背角神经元（Spinal Cord Neurons）与背根神经节（DRG）之间的细胞通讯网络（Sankey Diagram）。

## 🌐 在线访问
👉 **[点击直接访问在线应用](https://sc-visualize-knnahfbtdsdqdtpuj6q92b.streamlit.app/)**

---

## 📂 项目结构

```text
SC_project/
 data/                   # (本地忽略) 原始单细胞数据 (.h5ad, .h5)
 output/
    matrices/           # 可视化所需的核心矩阵数据 (.csv)
 scripts/
    config.py           # 全局路径与参数配置
    step1_...py         # 数据预处理流程 (Step 1-5)
    step6_visualization_app.py  # Streamlit 可视化启动脚本
 requirements.txt        # Python 依赖库列表
 README.md               # 项目说明文档
```

## 🚀 本地运行指南

如果您需要对代码进行修改或在本地运行，请遵循以下步骤：

### 1. 克隆仓库
```bash
git clone https://github.com/zhezixi180/sc-visualize.git
cd sc-visualize
```

### 2. 安装依赖环境
推荐使用 Python 3.10 环境以保证最佳兼容性。

```bash
# 创建虚拟环境 (可选)
conda create -n sc-viz python=3.10
conda activate sc-viz

# 安装依赖
pip install -r requirements.txt
```

### 3. 启动应用
在项目根目录下运行：

```bash
streamlit run scripts/step6_visualization_app.py
```
启动后，浏览器会自动打开 `http://localhost:8501`。

---

## 📊 数据处理流程 (Pipeline)

如果您拥有原始 `.h5ad` 数据并想重新生成矩阵，请按顺序运行 `scripts/` 下的步骤：

1.  **Step 1-3**: 数据读取、质控与聚类 (`make_dataset` -> `qc` -> `clustering`)
2.  **Step 4**: 细胞类型注释 (`annotate.py`)
3.  **Step 5**: 计算相互作用矩阵 (`interaction.py`) -> 生成 `output/matrices/` 下的 CSV 文件
4.  **Step 6**: 启动可视化 (`visualization_app.py`)

> 注意：为了仓库轻量化，原始 `.h5ad` 文件已被 `.gitignore` 忽略，仅上传了可视化所需的 CSV 结果。

## 🛠️ 常见问题 (FAQ)

**Q: 为什么应用提示 "App is sleeping"?**
A: 这是 Streamlit Cloud 的省电机制。若长时间无人访问应用会休眠，点击页面上的 **"Wake app up"** 按钮即可唤醒，通常需等待 1-2 分钟。

**Q: 线粒体阈值在哪里修改？**
A: 请查看 `scripts/config.py` 中的 `MAX_MITO_PCT` 参数。