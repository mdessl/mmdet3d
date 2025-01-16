import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Data from the table
modality_reduction = np.array([0, 10, 30, 50, 70, 90, 100])
bevfusion_camera = np.array([0.6858, 0.6769, 0.6609, 0.6463, 0.6310, 0.6183, 0.6131])
bevfusion_lidar = np.array([0.6856, 0.6088, 0.4646, 0.3109, 0.1596, 0.0251, 0.0082])
sbnet_camera = np.array([0.6294, 0.6205, 0.609, 0.6087, 0.6147, 0.6240, 0.6301])
sbnet_lidar = np.array([0.6294, 0.5730, 0.4462, 0.3247, 0.209, 0.1106, 0.0750])

# Professional color scheme
camera_color = '#2271B2'  # blue
lidar_color = '#D55E00'   # vermillion

# Create the plot with white background
plt.figure(figsize=(10, 6))
plt.style.use('default')

# Set white background
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Plot lines (without labels, we'll create custom legend)
plt.plot(modality_reduction, bevfusion_camera, color=camera_color, linestyle='--', marker='o', linewidth=2)
plt.plot(modality_reduction, bevfusion_lidar, color=lidar_color, linestyle='--', marker='o', linewidth=2)
plt.plot(modality_reduction, sbnet_camera, color=camera_color, linestyle='-', marker='s', linewidth=2)
plt.plot(modality_reduction, sbnet_lidar, color=lidar_color, linestyle='-', marker='s', linewidth=2)

# Create custom legend elements
legend_elements = [
    # Camera group
    Line2D([0], [0], color=camera_color, linestyle='--', marker='o', label='BEVFusion (% Camera Removed)', linewidth=2, markersize=1),
    Line2D([0], [0], color=camera_color, linestyle='-', marker='s', label='SBNet (% Camera Removed)', linewidth=2, markersize=1),
    # Separator
    Line2D([0], [0], color='none', label=''),
    # LiDAR group
    Line2D([0], [0], color=lidar_color, linestyle='--', marker='o', label='BEVFusion (% LiDAR Removed)', linewidth=2, markersize=1),
    Line2D([0], [0], color=lidar_color, linestyle='-', marker='s', label='SBNet (% LiDAR Removed)', linewidth=2, markersize=1)
]

# Customize the plot
plt.xlabel('% of Lidar/Camera Removed')
plt.ylabel('mAP')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(handles=legend_elements, frameon=True, facecolor='white', 
          edgecolor='none', bbox_to_anchor=(1.02, 1), loc='upper left')

# Set axis ranges
plt.xlim(-5, 105)
plt.ylim(0, 0.75)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save plot
plt.savefig('modality_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
