"""
Plotting utility functions and settings.
"""
import numpy as np

# plotting settings
FIGURE_WIDTH = 9# Nature Reviews maximum figure size: 180mm (w) x 215 mm (h)
FIGURE_HEIGHT = 13
PANEL_FONTSIZE = 17


def remove_spines(ax):
    """
    Remove the top and left spines from a matplotlib axis.
    """
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def smooth_interpolate(start_val, end_val, size): # code for this func borrowed from https://smustafaamir.medium.com/smoothsteps-simplified-12254adc773c and Google AI overview
    # 1. Create normalized array of points from 0 to 1
    t = np.linspace(0, 1, size)
    
    # 2. Smoothstep easing function: t^2 * (3 - 2t)
    # This creates an "ease in, ease out" curve with zero derivatives at endpoints
    t_smooth = t * t * (3 - 2 * t)
    
    # 3. Linearly interpolate between start and end
    return start_val + (end_val - start_val) * t_smooth