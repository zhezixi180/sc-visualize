
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import sys

# 引用公共配置
sys.path.append(str(Path(__file__).parent))
from config import *

def main():
    print("=== 7. Generate Sankey Plot ===")
    
    # 1. Load Data
    input_file = TABLES_DIR / "sankey_input_drg_spinal.csv"
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found. Run step6 first.")
        return
        
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} links.")
    
    # Top filtering (optional, if too messy)
    # df = df.head(50) 
    
    # 2. Prepare Nodes
    # Nodes are unique values from source and target columns
    all_nodes = list(pd.concat([df['source'], df['target']]).unique())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Map links to indices
    source_indices = df['source'].map(node_map)
    target_indices = df['target'].map(node_map)
    values = df['score']
    
    # 3. Define Colors (Optional)
    # Could map 'source' nodes to one color palette and 'target' to another
    
    # 4. Create Plot
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 15,
          thickness = 20,
          line = dict(color = "black", width = 0.5),
          label = all_nodes,
          color = "blue" # Default color, can be array
        ),
        link = dict(
          source = source_indices, # indices correspond to labels, eg A1, A2, A1, B1, ...
          target = target_indices,
          value = values
      ))])

    fig.update_layout(title_text="DRG to Spinal Cord Interactions (Pain/Itch)", font_size=10)
    
    # 5. Save
    output_html = FIGURES_DIR / "drg_spinal_sankey.html"
    fig.write_html(output_html)
    print(f"Saved interactive plot to {output_html}")
    
    # Try static save if kaleido/orca installed
    try:
        output_png = FIGURES_DIR / "drg_spinal_sankey.png"
        fig.write_image(output_png, scale=3)
        print(f"Saved static image to {output_png}")
    except Exception as e:
        print("Note: Static image export failed (requires kaleido). HTML is available.")

if __name__ == "__main__":
    main()
