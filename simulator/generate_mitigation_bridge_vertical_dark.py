import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import os

# Set up figure and axis (Vertical/Portrait)
fig, ax = plt.subplots(figsize=(5.5, 12), dpi=300)
fig.patch.set_facecolor('#121212') # Dark background
ax.set_facecolor('#121212')
ax.set_xlim(0, 5)
ax.set_ylim(0, 12)

# Definitions
rows = [
    {
        "left": "Sharp-onset disturbances evade detection",
        "right": "Layered with existing fast-acting MPC"
    },
    {
        "left": "Validated on synthetic data only",
        "right": "Retrainable directly on live DCS data"
    },
    {
        "left": "~71% precision — some false alarms",
        "right": "Accept/reject loop tunes threshold\nover time"
    },
    {
        "left": "Rate-constrained recommendations —\nmodest standalone impact",
        "right": "Paired with existing trajectory/\nrecipe tools"
    }
]

# Starting from top to bottom
y_starts = [10.5, 7.7, 4.9, 2.1]
box_width = 4.4
box_height = 0.8
center_x = 2.5

# Dark Theme Colors
left_box_color = '#450a0a'  # Dark red background for challenge
left_text_color = '#fecaca' # Light red text
right_box_color = '#052e16' # Dark green background for mitigation
right_text_color = '#bbf7d0'# Light green text
arrow_color = '#e5e5e5'     # Light grey arrow

for row, y_top in zip(rows, y_starts):
    y_bottom = y_top - 1.2
    
    # Top Box (Challenge)
    cx_y = y_top - box_height/2
    box_chal = patches.FancyBboxPatch((center_x - box_width/2, cx_y), box_width, box_height, 
                                      boxstyle="round,pad=0.1,rounding_size=0.2", 
                                      facecolor=left_box_color, edgecolor='#7f1d1d', linewidth=1.5)
    ax.add_patch(box_chal)
    
    # Bottom Box (Mitigation)
    mx_y = y_bottom - box_height/2
    box_mit = patches.FancyBboxPatch((center_x - box_width/2, mx_y), box_width, box_height, 
                                       boxstyle="round,pad=0.1,rounding_size=0.2", 
                                       facecolor=right_box_color, edgecolor='#14532d', linewidth=1.5)
    ax.add_patch(box_mit)
    
    # Arrow pointing DOWN
    arrow_start_y = y_top - box_height + 0.1
    arrow_end_y = y_bottom + 0.1
    ax.annotate('', xy=(center_x, arrow_end_y), xytext=(center_x, arrow_start_y),
                arrowprops=dict(arrowstyle="->", color=arrow_color, lw=2.5))
    
    # Text - Challenge
    left_text = "⚠️  " + row["left"]
    ax.text(center_x, y_top, left_text, ha='center', va='center', fontsize=12, color=left_text_color, fontweight='500', wrap=True)
    
    # Text - Mitigation
    right_text = "✔️  " + row["right"]
    ax.text(center_x, y_bottom, right_text, ha='center', va='center', fontsize=12, color=right_text_color, fontweight='500', wrap=True)


# Formatting
ax.axis('off')
plt.tight_layout()

# Save
output_path = r"C:\Users\nehau\Downloads\mitigation_bridge_vertical_dark.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='#121212', edgecolor='none')
print(f"Dark vertical mitigation bridge successfully generated and saved to {output_path}")
