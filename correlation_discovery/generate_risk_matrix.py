import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# Set axis limits
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

# Define the 4 quadrants with subtle, muted colors
# Bottom-Left (Low, Low) -> Light Green
ax.add_patch(patches.Rectangle((0, 0), 50, 50, linewidth=0, facecolor='#e2efda', alpha=1.0))
# Bottom-Right (High, Low) -> Light Yellow
ax.add_patch(patches.Rectangle((50, 0), 50, 50, linewidth=0, facecolor='#fff2cc', alpha=1.0))
# Top-Left (Low, High) -> Light Yellow
ax.add_patch(patches.Rectangle((0, 50), 50, 50, linewidth=0, facecolor='#fff2cc', alpha=1.0))
# Top-Right (High, High) -> Light Red/Orange
ax.add_patch(patches.Rectangle((50, 50), 50, 50, linewidth=0, facecolor='#fce4d6', alpha=1.0))

# Draw subtle gridlines at the 50% mark
ax.axhline(50, color='lightgrey', linestyle='--', linewidth=1.5)
ax.axvline(50, color='lightgrey', linestyle='--', linewidth=1.5)

# Define the risk points
risks = [
    {
        "x": 80, "y": 70,
        "label": "Sharp-onset disturbances\nevade detection",
        "align": "right", "offset": (-15, 0)
    },
    {
        "x": 50, "y": 80,
        "label": "Synthetic data vs.\nreal plant behavior",
        "align": "left", "offset": (15, 0)
    },
    {
        "x": 80, "y": 30,
        "label": "~71% precision\n(false alarms)",
        "align": "center", "offset": (0, -20)
    },
    {
        "x": 50, "y": 50,
        "label": "Rate-constrained\nrecommendations",
        "align": "left", "offset": (15, 15)
    }
]

# Plot points and text
for r in risks:
    ax.plot(r["x"], r["y"], marker='o', markersize=10, color='#2f5597', markeredgecolor='white', markeredgewidth=1.5)
    ax.annotate(r["label"], 
                (r["x"], r["y"]), 
                xytext=r["offset"], 
                textcoords='offset points',
                ha=r["align"], 
                va='center' if r["offset"][1] == 0 else 'bottom' if r["offset"][1] > 0 else 'top',
                fontsize=11, 
                color='#1f2d3d', 
                fontweight='500',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

# Format axes
ax.set_xticks([])
ax.set_yticks([])

# Add axis arrows and labels
# Y-axis (Impact)
ax.annotate('', xy=(0, 102), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1f2d3d", lw=2))
ax.text(-3, 50, 'Impact (Low \u2192 High)', rotation=90, va='center', ha='center', fontsize=14, fontweight='bold', color='#1f2d3d')

# X-axis (Likelihood)
ax.annotate('', xy=(102, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="#1f2d3d", lw=2))
ax.text(50, -5, 'Likelihood (Low \u2192 High)', va='center', ha='center', fontsize=14, fontweight='bold', color='#1f2d3d')

# Remove borders
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()

# Save image
output_path = r"C:\Users\nehau\Downloads\risk_matrix.png"
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Risk matrix successfully generated and saved to {output_path}")
