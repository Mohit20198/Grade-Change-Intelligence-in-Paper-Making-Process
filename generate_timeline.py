import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure
fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Timeline baseline
ax.annotate('', xy=(9.5, 5), xytext=(0.5, 5), arrowprops=dict(arrowstyle="->", color="#8c92ac", lw=3))
ax.text(9.6, 5, 'Time', va='center', fontsize=12, color='#8c92ac', fontweight='bold')

# Markers (T-5, T-2, T-0)
x_t5 = 1.8
x_t2 = 5.2
x_t0 = 8.5

# Tick marks
for x in [x_t5, x_t2, x_t0]:
    ax.plot([x, x], [4.8, 5.2], color='#8c92ac', lw=2)

# Marker 1: T-5 (Green)
circle1 = patches.Circle((x_t5, 5), radius=0.3, facecolor='#2ca02c', edgecolor='white', lw=2, zorder=3)
ax.add_patch(circle1)
ax.text(x_t5, 5, 'O', va='center', ha='center', color='white', fontsize=16, fontweight='bold', zorder=4) 
ax.text(x_t5, 5.8, "Early Warning\nSignal Detected", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1f2d3d')
ax.text(x_t5, 4.0, "Rising instability in\nstock_flow variance", ha='center', va='top', fontsize=10, color='#1f2d3d')
ax.text(x_t5, 4.6, "T-5 min", ha='center', va='top', fontsize=10, fontweight='bold', color='#2ca02c')

# Marker 2: T-2 (Blue)
circle2 = patches.Circle((x_t2, 5), radius=0.3, facecolor='#1f77b4', edgecolor='white', lw=2, zorder=3)
ax.add_patch(circle2)
ax.text(x_t2, 5, '!', va='center', ha='center', color='white', fontsize=18, fontweight='bold', zorder=4) 
ax.text(x_t2, 5.8, "Recommendation\nIssued", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#1f2d3d')
ax.text(x_t2, 4.0, "Rate-safe setpoint\nadjustment suggested", ha='center', va='top', fontsize=10, color='#1f2d3d')
ax.text(x_t2, 4.6, "T-2 min", ha='center', va='top', fontsize=10, fontweight='bold', color='#1f77b4')

# Marker 3: T-0 (Red, Faded)
circle3 = patches.Circle((x_t0, 5), radius=0.3, facecolor='#d62728', edgecolor='white', lw=2, zorder=3, alpha=0.3)
ax.add_patch(circle3)
ax.text(x_t0, 5, 'A', va='center', ha='center', color='white', fontsize=16, fontweight='bold', zorder=4, alpha=0.5) 
ax.text(x_t0, 5.8, "Off-Spec Breach\n(Avoided)", ha='center', va='bottom', fontsize=11, fontweight='bold', color='#d62728', alpha=0.5)
ax.text(x_t0, 4.0, "±2.5% deviation\nthreshold", ha='center', va='top', fontsize=10, color='#1f2d3d', alpha=0.5)
ax.text(x_t0, 4.6, "T-0", ha='center', va='top', fontsize=10, fontweight='bold', color='#d62728', alpha=0.5)

# Bracket/Arrow between 1 and 2
ax.annotate('', xy=(x_t5, 2.8), xytext=(x_t2, 2.8), arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2))
ax.text((x_t5+x_t2)/2, 2.6, "~3.5 min lead time", ha='center', va='top', fontsize=10, fontweight='bold', color='#2ca02c')

# Dashed grey arrow between 2 and 3
ax.annotate('', xy=(x_t0-0.4, 5), xytext=(x_t2+0.4, 5), arrowprops=dict(arrowstyle="->", color="darkgrey", lw=2, ls='dashed'))
ax.text((x_t2+x_t0)/2, 5.2, "if no action taken", ha='center', va='bottom', fontsize=9, color='darkgrey', style='italic')

# Branching arrow upward to "Stabilized"
x_stab = x_t2 + 1.2
y_stab = 8.0
# Draw a smooth curved arrow
ax.annotate('', xy=(x_stab, y_stab), xytext=(x_t2, 5.3),
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=2.5, connectionstyle="angle3,angleA=0,angleB=90"))
ax.text(x_stab+0.1, y_stab, "✓ Stabilized", va='center', ha='left', fontsize=13, fontweight='bold', color='#2ca02c')

plt.tight_layout()
output_path = r'C:\Users\nehau\Downloads\timeline.png'
plt.savefig(output_path, bbox_inches='tight', dpi=300, facecolor='white')
print(f"Timeline successfully generated and saved to {output_path}")
