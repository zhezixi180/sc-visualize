import os
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
#streamlit run d:\pythondaima\单细胞分析\project_2\SC_project\scripts\step6_visualization_app.py
# =========================================================
# 路径配置
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_DIR = PROJECT_ROOT / "output" / "matrices"

# =========================================================
# 常量定义 (DRG Mapping & Order)
# =========================================================
DRG_MAPPING = {
    # SNI 组 (优先)
    "Atf3/Gfra3/Gal": "SNIIC1",
    "Atf3/Mrgprd": "SNIIC2",
    "Atf3/S100b/Gal": "SNIIC3",
    # C 组
    "Cldn9": "C1-1",
    "Zcchc12/Sstr2": "C1-2-1",
    "Zcchc12/Dcn": "C1-2-2",
    "Zcchc12/Trpm8": "C1-2-3",
    "Zcchc12/Rxfp1": "C1-2-4",
    "Nppb": "C2",
    "Th/Fam19a4": "C3",
    "Mrgpra3": "C4-1",
    "Mrgpra3/Mrgprb4": "C4-2",
    "Mrgprd/Lpar3": "C5-1",
    "Mrgprd/Gm7271": "C5-2",
    "Wnt7a": "C7",
    "Trappc3l/Ntrk3/Gfra1": "C8-1",
    "Trappc3l/Prokr2": "C8-2",
    "Trappc3l/Smr2": "C8-3",
    "Baiap2l1": "C9"
}

# 严格的显示顺序 (Custom Order)
DRG_CUSTOM_ORDER = [
    "SNIIC1", "SNIIC2", "SNIIC3",
    "C1-1", "C1-2-1", "C1-2-2", "C1-2-3", "C1-2-4", 
    "C2", "C3", "C4-1", "C4-2", "C5-1", "C5-2",
    "C7", "C8-1", "C8-2", "C8-3", "C9"
]

def safe_read_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None

def safe_read_matrix_csv(path: Path) -> pd.DataFrame | None:
    """
    读取矩阵 CSV. 假设第一列是索引 (DRG 类型)，列名是 Spinal 类型。
    """
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0)

# =========================================================
# 加载矩阵数据
# =========================================================
@st.cache_data(show_spinner=False)
def load_available_mats():
    mats = {
        "pain": safe_read_matrix_csv(MATRIX_DIR / "pain_mat.csv"),
        "itch": safe_read_matrix_csv(MATRIX_DIR / "itch_mat.csv"),
        "modulatory": safe_read_matrix_csv(MATRIX_DIR / "modulatory_mat.csv"),
    }
    # 如果存在 synaptic 矩阵则加载 (预留接口)
    if (MATRIX_DIR / "synaptic_mat.csv").exists():
        mats["synaptic"] = safe_read_matrix_csv(MATRIX_DIR / "synaptic_mat.csv")
        
    layer_dist = safe_read_csv(MATRIX_DIR / "layer_dist.csv")
    lr_contrib = safe_read_csv(MATRIX_DIR / "lr_edge_contributions.csv")
    return mats, layer_dist, lr_contrib

# =========================================================
# 辅助函数
# =========================================================
def lamina_sort_key(lamina_name: str) -> tuple:
    """对层级名称进行排序 (Layer 1, Layer 2...)"""
    s = str(lamina_name).replace("Lamina:", "").replace("Layer", "").strip()
    # 尝试提取数字
    m = re.match(r"^(\d+)", s)
    if m:
        val = int(m.group(1))
        return (val, s)
    return (100, s) # 未知或其他放在最后

def _compute_even_y(n: int) -> list[float]:
    """计算 Sankey 图节点的均匀 Y 轴分布"""
    if n <= 0: return []
    if n == 1: return [0.5]
    # 尽可能撑满整个高度，从 0.005 到 0.995
    return [0.005 + 0.99 * (i / (n - 1)) for i in range(n)]

# =========================================================
# 核心逻辑: 矩阵 -> 连边
# =========================================================
def matrix_to_edges(
    mat: pd.DataFrame | None,
    threshold: float,
    top_n: int,
    normalize: bool = True
) -> pd.DataFrame:
    if mat is None:
        return pd.DataFrame(columns=["from", "to", "value"])

    m = mat.copy()

    # 可选: 再次执行全局最大值归一化 (用于可视化)
    # R 代码有时会在绘图前做归一化
    if normalize:
        mx = np.nanmax(m.values)
        if np.isfinite(mx) and mx > 0:
            m = m / mx

    # 展开矩阵为 [from, to, value] 格式
    # 修复 Pandas FutureWarning: The previous implementation of stack is deprecated...
    try:
        df = (
            m.stack(future_stack=True)
            .reset_index()
            .rename(columns={0: "value"})
        )
    except TypeError:
        # 兼容旧版 Pandas
        df = (
            m.stack(dropna=False)
            .reset_index()
            .rename(columns={0: "value"})
        )
    # 强制重命名列，防止索引名不匹配
    df.columns = ["from", "to", "value"]

    # 过滤无效值和微小值
    df = df[np.isfinite(df["value"]) & (df["value"] > 0.0001)] 
    
    # 应用用户阈值
    df = df[df["value"] > threshold]
    df = df.sort_values("value", ascending=False)

    if df.empty:
        return df

    # 取 Top N
    return df.head(int(min(int(top_n), len(df))))

# =========================================================
# 核心逻辑: Spinal细胞 -> 层级 (Lamina)
# =========================================================
def build_sc_to_lamina_edges(drg_sc: pd.DataFrame, layer_dist: pd.DataFrame | None) -> pd.DataFrame:
    """
    根据 DRG->SC 的流入量，结合 Spinal 细胞在各层级的分布频率，计算流向 Layer 的边。
    符合 R 代码逻辑中的流量守恒概念。
    """
    if drg_sc is None or drg_sc.empty or layer_dist is None or layer_dist.empty:
        return pd.DataFrame(columns=["from", "to", "value", "tooltip", "edge_group"])

    # 计算每个 Spinal 细胞类型从 DRG 接收的总流量 (Inflow)
    inflow = (
        drg_sc.groupby("to", as_index=False)["value"]
        .sum()
        .rename(columns={"to": "cell_type", "value": "inflow"})
    )

    # 校验 layer_dist 列名
    if "cell_type" not in layer_dist.columns or "layer" not in layer_dist.columns:
        st.error("layer_dist.csv 缺少必要列 (cell_type, layer)。")
        return pd.DataFrame()

    merged = layer_dist.merge(inflow, on="cell_type", how="inner")

        # ✅ 去掉缺失层信息（NaN 或字符串 nan）
    merged = merged[merged["layer"].notna()]
    merged = merged[merged["layer"].astype(str).str.strip().str.lower() != "nan"]
    
    # 流量分配: 该层的流量 = 总流入量 * 该层出现的频率
    merged["value"] = merged["inflow"] * merged["freq"]

    out = pd.DataFrame({
        "from": merged["cell_type"],
        "to": "Layer: " + merged["layer"].astype(str),
        "value": merged["value"],
        "tooltip": "",
        "edge_group": "lamina", # 标记为层级边，颜色区分
    })
    out = out[out["value"] > 0]
    return out

# =========================================================
# Tooltips (悬浮提示) & 生物学分组
# =========================================================
def build_drg_sc_with_tooltip(
    drg_sc: pd.DataFrame,
    lr_contrib: pd.DataFrame | None,
    top_k: int = 3,
    view_mode: str = "total",
) -> pd.DataFrame:
    drg_sc = drg_sc.copy()
    drg_sc["tooltip"] = ""
    drg_sc["edge_group"] = "unknown"
    drg_sc["edge_group_bio"] = "unknown"

    if drg_sc.empty or lr_contrib is None or lr_contrib.empty:
        return drg_sc

    # 仅筛选出当前图中存在的边
    key = drg_sc[["from", "to"]].drop_duplicates()
    contrib = lr_contrib.merge(key, left_on=["drg", "sc"], right_on=["from", "to"], how="inner")
    
    if contrib.empty:
        return drg_sc

    # 按 pathway/modality 聚合分数
    contrib2 = (
        contrib.groupby(["drg", "sc", "pathway", "modality"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "score"})
    )

    # 计算占比 (例如 Pain 占多少, Itch 占多少)
    contrib2["total"] = contrib2.groupby(["drg", "sc"])["score"].transform("sum")
    contrib2["frac"] = np.where(contrib2["total"] > 0, contrib2["score"] / contrib2["total"], 0.0)
    contrib2 = contrib2.sort_values(["drg", "sc", "score"], ascending=[True, True, False])
    
    # 仅展示 Top K 贡献的通路
    contrib_top = contrib2.groupby(["drg", "sc"], as_index=False).head(top_k)

    def pack_tooltip(g: pd.DataFrame) -> str:
        parts = []
        for _, r in g.iterrows():
            parts.append(f"{r['pathway']} ({r['modality']}): {r['frac']*100:.1f}%")
        return "<br/>".join(parts)

    tooltip_df = (
        contrib_top.groupby(["drg", "sc"])
        .apply(lambda g: pd.Series({"tooltip": pack_tooltip(g)}), include_groups=False)
        .reset_index()
    )

    def group_edge_bio(g: pd.DataFrame) -> str:
        mods = set(g["modality"].astype(str).tolist())
        if ("pain" in mods) and ("itch" in mods):
            return "mixed"
        if "pain" in mods:
            return "pain"
        if "itch" in mods:
            return "itch"
        if "modulatory" in mods:
            return "modulatory"
        return "unknown"

    bio_df = (
        contrib2.groupby(["drg", "sc"])
        .apply(lambda g: pd.Series({"edge_group_bio": group_edge_bio(g)}), include_groups=False)
        .reset_index()
    )

    # 合并回主表
    add = tooltip_df.merge(bio_df, on=["drg", "sc"], how="outer")
    out = drg_sc.merge(add, left_on=["from", "to"], right_on=["drg", "sc"], how="left")

    if "tooltip_y" in out.columns: out["tooltip"] = out["tooltip_y"].fillna("")
    if "edge_group_bio_y" in out.columns: out["edge_group_bio"] = out["edge_group_bio_y"].fillna("unknown")

    # 根据视图模式设置颜色分组
    if view_mode == "pain":
        out["edge_group"] = "pain"
    elif view_mode == "itch":
        out["edge_group"] = "itch"
    elif view_mode == "modulatory":
        # 强制设为 modulatory (绿色)，防止因 lookup 失败显示为灰色/黑色
        out["edge_group"] = "modulatory"
    elif view_mode == "mixed":
        out["edge_group"] = out["edge_group_bio"].astype(str)
    else:
        # Total 模式下，保持原有的 bio 分组逻辑
        out["edge_group"] = out["edge_group_bio"].astype(str)

    # 清理列
    cols_to_drop = ["drg", "sc", "tooltip_x", "tooltip_y", "edge_group_x", "edge_group_y", "edge_group_bio_x", "edge_group_bio_y"]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])

    return out

# =========================================================
# 构建 Sankey 对象
# =========================================================
def compute_stacked_y(node_indices: list, links: pd.DataFrame) -> list[float]:
    """
    计算基于权重的堆叠 Y 坐标，防止大节点重叠，同时保证填满高度。
    """
    if not node_indices:
        return []
        
    # 1. 计算每个节点的“权重” (连线总值)
    # Sankey 中节点高度由 max(inflow, outflow) 决定
    node_values = []
    for idx in node_indices:
        v_in = links.loc[links["target"] == idx, "value"].sum()
        v_out = links.loc[links["source"] == idx, "value"].sum()
        node_values.append(max(v_in, v_out))
        
    total_val = sum(node_values)
    n = len(node_indices)
    
    if total_val == 0 or n <= 1:
        if n <= 1: return [0.5]
        # 兜底：如果没权重，均匀分布
        return [0.01 + 0.98 * (i / (n - 1)) for i in range(n)]

    # 2. 设定 Gap 比例 
    # 显著增加 Gap 比例，确保节点之间有明显间隔 (用户要求 "Leave a gap")
    # 增加到 25% 的垂直空间用于留白
    gap_ratio = 0.25 
    content_ratio = 1.0 - gap_ratio
    
    # 每个节点分配的高度比例
    heights = [(v / total_val) * content_ratio for v in node_values]
    
    # 单个缝隙的高度 (均匀)
    gap_h = gap_ratio / (n - 1)
    
    # 3. 堆叠计算中心坐标
    # Plotly 中 y 是指的中心点
    y_centers = []
    current_y = 0.005 
    
    for h in heights:
        center = current_y + h / 2
        y_centers.append(center)
        current_y += h + gap_h
        
    return y_centers

# =========================================================
# 节点排序与颜色工具
# =========================================================
def get_node_ordering_by_weight(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """
    根据节点在网络中的总流量 (入度+出度) 进行排序 (降序)。
    """
    # 计算流量
    out_flow = edges.groupby("from")["value"].sum()
    in_flow = edges.groupby("to")["value"].sum()
    total_flow = out_flow.add(in_flow, fill_value=0)
    
    # 映射到 nodes
    nodes = nodes.copy()
    nodes["weight"] = nodes["name"].map(total_flow).fillna(0)
    
    return nodes

def get_drg_custom_sort_val(name: str) -> int:
    """获取 DRG 节点在自定义列表中的索引值"""
    # 先尝试直接匹配 Mapping 中的 Key
    # 如果 Mapping 中有，转换成 Short Name，再查 Order
    short_name = DRG_MAPPING.get(name)
    if short_name and short_name in DRG_CUSTOM_ORDER:
        return DRG_CUSTOM_ORDER.index(short_name)
    
    # 如果没找到，返回一个很大的数，排在最后
    return 9999

def format_drg_label(name: str) -> str:
    """将 DRG 名称转换为 '简称(原始名)' 格式"""
    short = DRG_MAPPING.get(name)
    if short:
        return f"{short}({name})"
    return name

def build_sankey(drg_sc: pd.DataFrame, sc_lamina: pd.DataFrame, hide_isolated: bool = True, sort_by: str = "Weight"):
    edges = pd.concat(
        [
            drg_sc.assign(edge_type="DRG→SC"),
            sc_lamina.assign(edge_type="SC→Lamina"),
        ],
        ignore_index=True,
    )

    if edges.empty:
        return pd.DataFrame(), pd.DataFrame()

    # 1. 初始构建完整节点表 (带分类标记)
    # 左: DRG
    drg_nodes_list = sorted(edges.loc[edges["edge_type"] == "DRG→SC", "from"].astype(str).unique().tolist())
    
    # 中: SC
    sc_nodes_list = sorted(pd.concat([
        edges.loc[edges["edge_type"] == "DRG→SC", "to"],
        edges.loc[edges["edge_type"] == "SC→Lamina", "from"]
    ]).dropna().astype(str).unique().tolist())
    
    # 右: Lamina (使用专门的排序键)
    lamina_nodes_list = sorted(
        edges.loc[edges["edge_type"] == "SC→Lamina", "to"].dropna().astype(str).unique().tolist(),
        key=lamina_sort_key
    )

    # 创建带类型的子表
    df_drg = pd.DataFrame({"name": drg_nodes_list, "label": drg_nodes_list, "type": "drg"})
    
    # 应用 DRG 标签重命名 (如果是 DRG 节点，且有 Mapping)
    df_drg["label"] = df_drg["name"].apply(format_drg_label)

    df_sc = pd.DataFrame({"name": sc_nodes_list, "label": sc_nodes_list, "type": "sc"})
    df_lam = pd.DataFrame({"name": lamina_nodes_list, "label": lamina_nodes_list, "type": "lamina"})
    # Lamina 始终保持层级顺序 (Layer 1 -> 2 -> 3)
    df_lam["order_val"] = range(len(df_lam)) 

    nodes = pd.concat([df_drg, df_sc, df_lam], ignore_index=True)

    # 建立临时索引映射
    idx_map = {name: i for i, name in enumerate(nodes["name"])}
    
    links = edges.copy()
    links["source"] = links["from"].astype(str).map(idx_map)
    links["target"] = links["to"].astype(str).map(idx_map)
    links = links.dropna(subset=["source", "target", "value"])

    # 2. 移除孤立节点
    if hide_isolated and not links.empty:
        used_indices = set(links["source"]).union(set(links["target"]))
        nodes = nodes.iloc[list(used_indices)]

    # 3. 重组最终节点表
    # 先计算默认权重 (任何模式下可能都需要用来处理 '其他' 节点)
    nodes = get_node_ordering_by_weight(nodes, edges)

    if sort_by == "Custom (Paper)":
        # DRG: 自定义顺序 (未定义的排最后并按权重降序)
        nodes["custom_order"] = nodes["name"].apply(get_drg_custom_sort_val)
        
        drg_part = nodes[nodes["type"] == "drg"].sort_values(
            by=["custom_order", "weight"], 
            ascending=[True, False]
        )
        
        # SC: 仍然按权重
        sc_part = nodes[nodes["type"] == "sc"].sort_values("weight", ascending=False)
        
        nodes_final = pd.concat([
            drg_part,
            sc_part,
            nodes[nodes["type"] == "lamina"].sort_values("order_val")
        ], ignore_index=True).reset_index(drop=True)
        
    elif sort_by == "Weight":
        # DRG/SC 按权重降序，Lamina 仍按 OrderVal
        nodes_final = pd.concat([
            nodes[nodes["type"] == "drg"].sort_values("weight", ascending=False),
            nodes[nodes["type"] == "sc"].sort_values("weight", ascending=False),
            nodes[nodes["type"] == "lamina"].sort_values("order_val")
        ], ignore_index=True).reset_index(drop=True)
    else:
        # 默认按名称排序 ( fallback )
        nodes_final = pd.concat([
            nodes[nodes["type"] == "drg"].sort_values("name"),
            nodes[nodes["type"] == "sc"].sort_values("name"),
            nodes[nodes["type"] == "lamina"].sort_values("order_val")
        ], ignore_index=True).reset_index(drop=True)

    # 4. 更新 Links 指向新的节点索引
    final_idx_map = {name: i for i, name in enumerate(nodes_final["name"])}
    links["source"] = links["from"].astype(str).map(final_idx_map)
    links["target"] = links["to"].astype(str).map(final_idx_map)
    links = links.dropna(subset=["source", "target"])

    # 5. 计算堆叠 Y 坐标 (核心修正: 基于权重分布而非均匀分布)
    # 这能保证大节点占用更多空间，小节点占用更少，避免重叠
    
    # DRG
    drg_indices = nodes_final[nodes_final["type"] == "drg"].index.tolist()
    y_drg = compute_stacked_y(drg_indices, links)
    
    # SC
    sc_indices = nodes_final[nodes_final["type"] == "sc"].index.tolist()
    y_sc = compute_stacked_y(sc_indices, links)
    
    # Lamina
    lam_indices = nodes_final[nodes_final["type"] == "lamina"].index.tolist()
    y_lam = compute_stacked_y(lam_indices, links)
    
    # 分配坐标
    nodes_final["x"] = 0.0
    nodes_final["y"] = 0.0
    
    nodes_final.loc[drg_indices, "x"] = 0.01
    nodes_final.loc[drg_indices, "y"] = y_drg
    
    nodes_final.loc[sc_indices, "x"] = 0.50
    nodes_final.loc[sc_indices, "y"] = y_sc
    
    nodes_final.loc[lam_indices, "x"] = 0.99
    nodes_final.loc[lam_indices, "y"] = y_lam

    return nodes_final, links

# =========================================================
# Plotly 渲染器
# =========================================================
EDGE_COLOR = {
    "pain": "rgba(220,20,60,0.55)",
    "itch": "rgba(30,144,255,0.55)",
    "modulatory": "rgba(34,139,34,0.55)",
    "mixed": "rgba(138,43,226,0.55)",
    "lamina": "rgba(169,169,169,0.40)", # 改为标准的深灰色 (DarkGray)，透明度稍高，使其更像灰色
    "unknown": "rgba(0,0,0,0.15)",      # 改为纯黑但极低透明度，使其在此背景下视觉上更偏“淡黑/阴影”
}

def render_sankey_plotly(nodes, links, selected_edge: dict | None = None):
    if nodes.empty or links.empty:
        st.write("数据不足，无法绘制 Sankey 图。")
        return

    # 分配颜色
    # 如果有高亮选中边，则高亮边为黑色，其他为浅灰色
    colors = []
    
    # 将 selected_edge 转换为字符串以便比较 (防止类型不匹配)
    sel_from = str(selected_edge.get("from", "")) if selected_edge else ""
    sel_to = str(selected_edge.get("to", "")) if selected_edge else ""
    sel_type = str(selected_edge.get("edge_type", "")) if selected_edge else ""

    for _, row in links.iterrows():
        # 如果有选中项，检查是否匹配
        is_selected = False
        if selected_edge:
            # 简单匹配 from 和 to
            if str(row["from"]) == sel_from and str(row["to"]) == sel_to:
                # 还可以加 edge_type 校验更严谨，但这里 from->to 应该够了
                is_selected = True

        if selected_edge:
            if is_selected:
                colors.append("rgba(0, 0, 0, 0.90)") # 高亮黑
            else:
                colors.append("rgba(180, 180, 180, 0.15)") # 极淡灰 (非选中)
        else:
            # 默认模式：按组着色
            g = str(row.get("edge_group", "unknown"))
            colors.append(EDGE_COLOR.get(g, EDGE_COLOR["unknown"]))

    # 动态计算高度，防止节点过密被截断
    # 获取各列节点数
    n_drg = len(nodes[nodes["x"] < 0.1])
    n_sc = len(nodes[(nodes["x"] > 0.4) & (nodes["x"] < 0.6)])
    n_lam = len(nodes[nodes["x"] > 0.9])
    max_nodes = max(n_drg, n_sc, n_lam)
    
    # 每个节点预留高度: 节点越多，给的越少以防图过大，但有下限
    # 稍微增加行高配合大 Gap，防止小节点消失
    row_height = 45 
        
    dynamic_height = max(750, max_nodes * row_height)
    
    # 动态调整节点间距 (Padding) - 仅作为 Plotly 参考，实际位置由 Y 决定
    dynamic_pad = 15

    # 构造 Custom Data 用于悬浮
    customdata = []
    for _, row in links.iterrows():
        tip = row.get("tooltip", "")
        if isinstance(tip, pd.Series): tip = tip.iloc[0] if not tip.empty else ""
        val = row.get("value", 0)
        customdata.append(f"<b>{row['from']}</b> → <b>{row['to']}</b><br>Value: {val:.4g}<br>{tip}")

    # 生成节点颜色 (高对比度，确保相邻节点颜色区分明显)
    # 使用 Hash 映射确保颜色一致性
    import hashlib

    palette_hex = [
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
        "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
        "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
        "#17becf", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39"
    ]
    # 对色盘进行切片重组，进一步打乱连续色系的排列（例如把深蓝和浅蓝分开）
    mixed_palette = palette_hex[::2] + palette_hex[1::2] 
    
    def get_stable_color(text: str) -> str:
        """根据文本内容的 MD5 哈希值分配固定颜色"""
        if not isinstance(text, str): text = str(text)
        h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)
        return mixed_palette[h % len(mixed_palette)]

    node_colors = [get_stable_color(name) for name in nodes["name"]]

    fig = go.Figure(data=[go.Sankey(
        arrangement = "fixed", 
        node = dict(
            pad = dynamic_pad,
            thickness = 15,
            line = dict(color = "black", width = 0.5),
            label = nodes["label"],
            x = nodes["x"],
            y = nodes["y"],
            color = node_colors # 应用多彩颜色
        ),
        link = dict(
            source = links["source"],
            target = links["target"],
            value = links["value"],
            color = colors,
            customdata = customdata,
            hovertemplate = '%{customdata}<extra></extra>' 
        )
    )])
    
    fig.update_layout(
        title_text=None,
        font_size=12, 
        height=dynamic_height,
        # 大幅增加底部边距，防止 Layer 最后一个标签被截断
        margin=dict(l=20, r=20, t=20, b=150) 
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================================================
# 主程序
# =========================================================
def main():
    st.set_page_config(layout="wide", page_title="DRG -> Spinal -> Layer")
    st.title("细胞交互网络: DRG -> Spinal Cell -> Layer")
    
    mats, layer_dist, lr_contrib = load_available_mats()
    
    with st.sidebar:
        st.header("控制面板")
        
        # 模式选择
        mode = st.radio("通路模式 (Pathway Mode)", ["total", "pain", "itch", "modulatory", "mixed"], index=0)
        
        # 矩阵合并逻辑
        mat = None
        if mode == "total":
            # 总和 pain + itch + modulatory
            dfs = [mats[k] for k in ["pain", "itch", "modulatory"] if mats[k] is not None]
            if dfs:
                mat = dfs[0].copy()
                for d in dfs[1:]: mat = mat.add(d, fill_value=0)
        elif mode == "mixed":
            # 同样先取总量，后面再按 bio group 过滤
            dfs = [mats[k] for k in ["pain", "itch", "modulatory"] if mats[k] is not None]
            if dfs:
                mat = dfs[0].copy()
                for d in dfs[1:]: mat = mat.add(d, fill_value=0)
        else:
            mat = mats.get(mode)
            
        if mat is None:
            st.error(f"该模式 {mode} 没有找到对应的数据矩阵。请先运行 step5。")
            st.stop()
            
        # 阈值控制
        max_val = float(np.nanmax(mat.values)) if not mat.empty else 1.0
        threshold = st.slider("互作分数阈值 (Threshold)", 0.0, max_val, 0.0, max_val/100)
        top_n = st.number_input("最大边数显示 (Top N Edges)", 10, 5000, 300)
        
        hide_iso = st.checkbox("隐藏孤立节点", True)
        
        sort_mode = st.radio("节点排序 (Sort Nodes)", ["Custom (Paper)", "Weight"], index=0)
        
        st.divider()
        st.caption("连线颜色说明:")
        st.caption("红=Pain, 蓝=Itch, 绿=Modulatory")
        st.caption("紫=Mixed")
        st.caption("灰(右)=Lamina (解剖层级分布)")
        st.caption("黑/灰(左)=Unknown (数据库中未注明的互作)")
        st.caption("注意：节点颜色仅用于视觉区分 (Total包含上述所有)")

        if st.button("清除高亮选择"):
            st.session_state["selected_edge"] = None
            if "edge_table" in st.session_state:
                del st.session_state["edge_table"]

    # --- 顶部筛选 (Top Filters) ---
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        drg_options = sorted(mat.index.tolist())
        drg_keep = st.multiselect("DRG 类群 (可选)", options=drg_options, default=[])
    
    with col_filter2:
        sc_options = sorted(mat.columns.tolist())
        sc_keep = st.multiselect("SC 类群 (可选)", options=sc_options, default=[])

    # 1. 矩阵 -> 边
    drg_sc = matrix_to_edges(mat, threshold, top_n)
    
    # 应用顶部筛选
    if drg_keep:
        drg_sc = drg_sc[drg_sc["from"].isin(drg_keep)]
    if sc_keep:
        drg_sc = drg_sc[drg_sc["to"].isin(sc_keep)]
    
    # 2. 丰富信息 (Tooltip, Bio Group)
    drg_sc = build_drg_sc_with_tooltip(drg_sc, lr_contrib, view_mode=mode)
    
    # 3. 如果是 Mixed 模式，过滤非 Mixed 的边
    if mode == "mixed":
        drg_sc = drg_sc[drg_sc["edge_group_bio"] == "mixed"]
        
    # 4. 构建脊髓 -> 层级边
    sc_lamina = build_sc_to_lamina_edges(drg_sc, layer_dist)
    
    # 5. 生成 Sankey 数据
    nodes, links = build_sankey(drg_sc, sc_lamina, hide_isolated=hide_iso, sort_by=sort_mode)
    
    # --- 布局分栏: 左图 右表 ---
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        # 获取当前选中的边状态
        selected = st.session_state.get("selected_edge", None)
        render_sankey_plotly(nodes, links, selected_edge=selected)
    
    with col_right:
        st.subheader("边表 (筛选后按 value 降序)")
        
        if links.empty:
            st.write("暂无数据")
        else:
            # 整理数据
            show_df = links.copy()
            show_df["tooltip"] = show_df["tooltip"].apply(lambda x: str(x) if pd.notnull(x) else "")
            
            # --- 右侧高级筛选控件 ---
            
            # 1. 搜索
            search_term = st.text_input("搜索 (过滤 from/to, 支持子串匹配)", "")
            if search_term:
                mask = (
                    show_df["from"].astype(str).str.contains(search_term, case=False) | 
                    show_df["to"].astype(str).str.contains(search_term, case=False)
                )
                show_df = show_df[mask]

            # 2. 连线类型筛选
            edge_type_opts = ["全部", "DRG→SC", "SC→Lamina"]
            edge_type_choice = st.selectbox("只看哪一段连线", edge_type_opts, index=0)
            if edge_type_choice != "全部":
                show_df = show_df[show_df["edge_type"] == edge_type_choice]

            # 3. 分组筛选
            all_groups = show_df["edge_group"].astype(str).unique().tolist()
            # 常用排序
            group_priority = ["pain", "itch", "mixed", "modulatory", "lamina", "unknown"]
            ordered_groups = [g for g in group_priority if g in all_groups] + [g for g in all_groups if g not in group_priority]
            
            group_choice = st.multiselect(
                "只看哪些分组 (对应连线颜色/类别)",
                options=ordered_groups,
                default=ordered_groups
            )
            if group_choice:
                show_df = show_df[show_df["edge_group"].isin(group_choice)]
            else:
                show_df = show_df.iloc[0:0] # 全不选则清空

            # 4. Top K
            only_topk = st.checkbox("仅显示 Top K (防止表太大)", value=False)
            filter_k = st.number_input("K", 50, 10000, 1000, disabled=not only_topk)
            if only_topk:
                show_df = show_df.head(int(filter_k))

            # 5. 下载按钮区
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                # 仅筛选 DRG->SC 部分用于下载
                dl_drg = show_df[show_df["edge_type"] == "DRG→SC"]
                st.download_button(
                    "下载 DRG→SC (按当前筛选) CSV",
                    data=dl_drg.to_csv(index=False).encode("utf-8"),
                    file_name=f"drg_sc_filtered_{mode}.csv",
                    mime="text/csv"
                )
            with d_col2:
                # 仅筛选 SC->Lamina 部分用于下载
                dl_lam = show_df[show_df["edge_type"] == "SC→Lamina"]
                st.download_button(
                    "下载 SC→Lamina (按当前筛选) CSV",
                    data=dl_lam.to_csv(index=False).encode("utf-8"),
                    file_name=f"sc_lamina_filtered_{mode}.csv",
                    mime="text/csv"
                )

            st.caption(f"当前显示 {len(show_df)} 条边 (筛选后)")

            # 仅展示关键列
            display_cols = ["edge_type", "from", "to", "value", "edge_group", "tooltip"]
            display_cols = [c for c in display_cols if c in show_df.columns]
            
            # 定义选中回调
            def on_table_select():
                sel_state = st.session_state.get("edge_table", {})
                selection = sel_state.get("selection", {})
                rows = selection.get("rows", [])
                if rows:
                    idx = rows[0]
                    if 0 <= idx < len(show_df):
                        row_data = show_df.iloc[idx]
                        st.session_state["selected_edge"] = {
                            "from": row_data["from"],
                            "to": row_data["to"],
                            "edge_type": row_data.get("edge_type", "")
                        }
                else:
                    st.session_state["selected_edge"] = None

            st.dataframe(
                show_df[display_cols],
                use_container_width=True,
                height=700,
                hide_index=True,
                on_select=on_table_select,
                selection_mode="single-row",
                key="edge_table"
            )

    # 增加额外空行
    st.markdown("<br>" * 2, unsafe_allow_html=True)

    # 数据展示 (用户要求暂时注释掉)
    # with st.expander("查看原始全量数据"):
    #     st.dataframe(links)

if __name__ == "__main__":
    main()
