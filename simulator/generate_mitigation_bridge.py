import matplotlib.pyplot as plt
import matplotlib.patches as patches
import textwrap
import os

# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
ax.set_xlim(0, 11)
ax.set_ylim(0, 5.5)

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

y_positions = [4, 3, 2, 1]
box_width = 4.2
box_height = 0.6
left_center_x = 2.8
right_center_x = 8.2

left_color = '#fce4d6'  # light red/orange
right_color = '#e2efda' # light green
text_color = '#1f2d3d'  # dark navy/charcoal

# Draw Headers
ax.text(left_center_x, 5.0, "Challenge", ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)
ax.text(right_center_x, 5.0, "Mitigation Strategy", ha='center', va='center', fontsize=16, fontweight='bold', color=text_color)

# Draw Rows
for i, (row, y) in enumerate(zip(rows, y_positions)):
    
    # Left Box (Challenge)
    lx = left_center_x - box_width/2
    ly = y - box_height/2
    box_left = patches.FancyBboxPatch((lx, ly), box_width, box_height, 
                                      boxstyle="round,pad=0.1,rounding_size=0.2", 
                                      facecolor=left_color, edgecolor='none')
    ax.add_patch(box_left)
    
    # Right Box (Mitigation)
    rx = right_center_x - box_width/2
    ry = y - box_height/2
    box_right = patches.FancyBboxPatch((rx, ry), box_width, box_height, 
                                       boxstyle="round,pad=0.1,rounding_size=0.2", 
                                       facecolor=right_color, edgecolor='none')
    ax.add_patch(box_right)
    
    # Arrow
    arrow_start_x = left_center_x + box_width/2 + 0.1
    arrow_end_x = right_center_x - box_width/2 - 0.1
    ax.annotate('', xy=(arrow_end_x, y), xytext=(arrow_start_x, y),
                arrowprops=dict(arrowstyle="->", color=text_color, lw=2.5))
    
    # Text - Left
    # Add a small warning emoji space
    left_text = "⚠️  " + row["left"]
    ax.text(left_center_x, y, left_text, ha='center', va='center', fontsize=11, color=text_color, fontweight='500', wrap=True)
    
    # Text - Right
    # Add a small checkmark emoji space
    right_text = "✔️  " + row["right"]
    ax.text(right_center_x, y, right_text, ha='center', va='center', fontsize=11, color=text_color, fontweight='500', wrap=True)


# Formatting
ax.axis('off')
plt.tight_layout()

# Save
output_path = r"C:\Users\nehau\Downloads\mitigation_bridge.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Mitigation bridge successfully generated and saved to {output_path}")
