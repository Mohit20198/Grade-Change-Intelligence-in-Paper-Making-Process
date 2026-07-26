import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap

# Set up figure and axis to roughly a 6:5 ratio (e.g. 4.2 x 3.5 inches)
fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Tile contents
tiles = [
    {"val": "5/5", "lbl": "Ground-Truth Causal Links Recovered"},
    {"val": "0.9838", "lbl": "Classifier AUC"},
    {"val": "3.5 min", "lbl": "Mean Early-Warning Lead Time"},
    {"val": "6/9", "lbl": "Events with Early Warning"},
    {"val": "16", "lbl": "Spurious Correlations Correctly Rejected"},
    {"val": "71%", "lbl": "Precision (Recall: 83%)"}
]

# Layout parameters
box_w = 0.42
box_h = 0.28
x_centers = [0.27, 0.73]
y_centers = [0.82, 0.50, 0.18]

# Colors
tile_bg = '#f1f5f9'      # Light slate/grey background for tiles
val_color = '#0f172a'    # Dark navy/slate for big numbers
lbl_color = '#475569'    # Medium grey for labels

# Draw Grid
for i, tile in enumerate(tiles):
    # Determine row and col
    r = i // 2
    c = i % 2
    cx, cy = x_centers[c], y_centers[r]
    
    # Draw tile background
    box = patches.FancyBboxPatch((cx - box_w/2, cy - box_h/2), box_w, box_h, 
                                 boxstyle="round,pad=0.02,rounding_size=0.08", 
                                 facecolor=tile_bg, edgecolor='none')
    ax.add_patch(box)
    
    # Draw Big Number
    ax.text(cx, cy + 0.04, tile["val"], ha='center', va='center', 
            fontsize=20, fontweight='bold', color=val_color)
    
    # Draw Label (wrapped)
    wrapped_lbl = textwrap.fill(tile["lbl"], width=22)
    ax.text(cx, cy - 0.06, wrapped_lbl, ha='center', va='center', 
            fontsize=7, fontweight='500', color=lbl_color, linespacing=1.4)

ax.axis('off')
plt.tight_layout(pad=0.5)

# Save
output_path = r"C:\Users\nehau\Downloads\results_metrics_strip.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white', edgecolor='none')
print(f"Metrics strip successfully generated and saved to {output_path}")
