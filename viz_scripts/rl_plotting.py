import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # makes math look Times-like too

# w/o health feedback with health feedback all datahealth-based episode filteringdamage-based data fitlering

color_maps = {
    'RL (w/o health feedback)': 'C0',
    'RL (with health feedback)': 'C1',
}

alpha_map = {
    'RL (w/o health feedback)': 0.7,
    'RL (with health feedback)': 0.7,
}
method_list = ['RL (w/o health feedback)', 'RL (with health feedback)']

# # move_glass_of_water
# color_maps = {
#     'BC-GMM': 'C0',
#     'RL Finetune': 'C1',
# }

# alpha_map = {
#     'BC-GMM': 0.7,
#     'RL Finetune': 0.7,
# }
# method_list = ['BC-GMM', 'RL Finetune']

# # shelve_cereal_box
# color_maps = {
#     'BC-flow': 'C0',
#     'DSRL Finetune': 'C1',
# }

# alpha_map = {
#     'BC-flow': 0.7,
#     'DSRL Finetune': 0.7,
# }
# method_list = ['BC-flow', 'DSRL Finetune']


_real_results = {
    'place_plate': {
        'full': {
                'RL (w/o health feedback)': {'task_completion_rate': 0.8, 'safe_task_completion_rate': 0.01},
                'RL (with health feedback)': {'task_completion_rate': 0.9, 'safe_task_completion_rate': 0.86},
            }
    },
    # 'move_glass_of_water': {
    #     'full': {
    #             'BC-GMM': {'task_completion_rate': 1.0, 'safe_task_completion_rate': 0.2},
    #             'RL Finetune': {'task_completion_rate': 1.0, 'safe_task_completion_rate': 1.0},
    #         }
    # },
    # 'shelve_cereal_box': {
    #     'full': {
    #             'BC-flow': {'task_completion_rate': 0.73, 'safe_task_completion_rate': 0.13},
    #             'DSRL Finetune': {'task_completion_rate': 0.66, 'safe_task_completion_rate': 0.46},
    #         }
    # },
}


def lighten_color(color, factor=0.3):
    """Lighten a color by a factor (0-1). Factor closer to 1 = lighter."""
    if isinstance(color, str):
        # Handle matplotlib color strings
        import matplotlib.colors as mcolors
        color = mcolors.to_rgb(color)
    return tuple(min(1.0, c + (1 - c) * factor) for c in color)

def darken_color(color, factor=0.3):
    """Darken a color by a factor (0-1). Factor closer to 1 = darker."""
    if isinstance(color, str):
        # Handle matplotlib color strings
        import matplotlib.colors as mcolors
        color = mcolors.to_rgb(color)
    return tuple(c * (1 - factor) for c in color)

def plot_real_task(results, plot_partial=False, plot_legend=False, linewidth=0, 
                   metric1_key='Task Completion Rate', metric2_key='Safe Task Completion Rate', 
                   metric1_results=None, metric2_results=None):
    """
    Plot bar chart with two columns per method.
    
    Args:
        results: Dictionary with 'full' key containing method results (for backward compatibility)
        metric1_key: Key for first metric if results['full'][method] is a dict
        metric2_key: Key for second metric if results['full'][method] is a dict
        metric1_results: Optional dict of {method: value} for first metric (overrides results)
        metric2_results: Optional dict of {method: value} for second metric (overrides results)
    """
    if plot_partial:
       pass

    # Determine which methods to plot and get metric values
    if metric1_results is not None and metric2_results is not None:
        # Use provided metric dictionaries
        keys = list(metric1_results.keys())
        metric1_vals = [metric1_results[k] for k in keys]
        metric2_vals = [metric2_results[k] for k in keys]
    else:
        # Use results structure
        full_results = results['full']
        keys = list(full_results.keys())
        metric1_vals = []
        metric2_vals = []
        for k in keys:
            if isinstance(full_results[k], dict):
                metric1_vals.append(full_results[k].get(metric1_key, 0))
                metric2_vals.append(full_results[k].get(metric2_key, 0))
            else:
                # Backward compatibility: if it's a single value, use it for both
                metric1_vals.append(full_results[k])
                metric2_vals.append(full_results[k])

    # plot a bar graph with the data and show the labels in a legend
    fig, ax = plt.subplots(figsize=(3, 5))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Bar width for each individual bar
    bar_width = 0.02
    # Gap between the two bars of the same method (0 = touching)
    gap_within_method = 0.0
    # Gap between different methods
    gap_between_methods = 0.02
    
    # Calculate positions
    # Using align='edge' so bars start at x position (not centered)
    x_positions = []
    current_x = 0
    
    for i, k in enumerate(keys):
        # Position for first bar (metric1) - left bar
        x1 = current_x
        # Position for second bar (metric2) - right bar (touching the first)
        x2 = current_x + bar_width + gap_within_method
        x_positions.append((x1, x2))
        # Move to next method group position
        # Right edge of second bar is at (x2 + bar_width), add gap for next method
        current_x = x2 + bar_width + gap_between_methods
    
    # Get colors and create patterns
    # Hatch patterns: metric1 = solid (no pattern), metric2 = diagonal lines
    hatch1 = ''  # No pattern for metric1 (solid)
    hatch2 = '///'  # Diagonal lines for metric2
    
    for i, k in enumerate(keys):
        base_color = color_maps[k]
        base_alpha = alpha_map[k]
        
        x1, x2 = x_positions[i]
        val1 = metric1_vals[i]
        val2 = metric2_vals[i]
        
        # Plot bars - label method only once (on first bar)
        method_label = k if i == 0 else ''
        
        if linewidth > 0:
            ax.bar(x1, val1, bar_width, label=method_label, align='edge',
                   color=base_color, alpha=base_alpha, edgecolor='black', 
                   linewidth=linewidth, hatch=hatch1)
            ax.bar(x2, val2, bar_width, label='', align='edge',
                   color=base_color, alpha=base_alpha, edgecolor='black', 
                   linewidth=linewidth, hatch=hatch2)
        else:
            ax.bar(x1, val1, bar_width, label=method_label, align='edge',
                   color=base_color, alpha=base_alpha, edgecolor='black', 
                   linewidth=0.5, hatch=hatch1)
            ax.bar(x2, val2, bar_width, label='', align='edge',
                   color=base_color, alpha=base_alpha, edgecolor='black', 
                   linewidth=0.5, hatch=hatch2)

    # Set up axes
    # Calculate center positions for each method group (between the two bars)
    method_centers = [(x_positions[i][0] + x_positions[i][1] + bar_width) / 2 for i in range(len(keys))]
    ax.set_xticks(method_centers)
    # Add line breaks to x-axis labels (break after "w/o" or "with")
    x_labels = []
    for k in keys:
        if 'w/o ' in k:
            label = k.replace('w/o ', 'w/o\n')
        elif 'with ' in k:
            label = k.replace('with ', 'with\n')
        else:
            label = k
        x_labels.append(label)
    ax.set_xticklabels(x_labels, fontsize=11, rotation=0, ha='center')
    plt.ylim([0, 100])
    plt.yticks(np.arange(0, 101, 20), fontsize=14)
    
    # Show all spines (axes borders)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['top'].set_color('black')
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3, color='gray')
    ax.set_axisbelow(True)  # Grid behind bars
    if plot_legend:
        # Create custom legend with metric patterns only (methods shown on x-axis)
        import matplotlib.patches as mpatches
        
        # Create pattern legend entries for metrics (using white color to show pattern clearly)
        pattern1_patch = mpatches.Patch(facecolor='white', edgecolor='black', 
                                        hatch=hatch1, label=metric1_key)
        pattern2_patch = mpatches.Patch(facecolor='white', edgecolor='black', 
                                        hatch=hatch2, label=metric2_key)
        
        plt.legend([pattern1_patch, pattern2_patch], [metric1_key, metric2_key], 
                  loc='upper left', fontsize=12, frameon=False, ncol=2, columnspacing=1, 
                  bbox_to_anchor=(1.15, 0.8))
    plt.tight_layout()
    return ax


if __name__ == "__main__":
    real_results = {}
    metrics = ['task_completion_rate', 'safe_task_completion_rate']
    for task in _real_results:
        real_results[task] = {}
        for part in _real_results[task]:
            real_results[task][part] = {}
            for method in method_list:
                real_results[task][part][method] = {}
                for metric in metrics:
                    real_results[task][part][method][metric] = _real_results[task][part][method][metric] * 100

    target_task = 'place_plate'
    
    # Example 1: Using separate metric dictionaries (recommended)
    # Create two dictionaries, one for each metric
    metric1_data = {method: real_results[target_task]['full'][method]["task_completion_rate"] for method in method_list}
    metric2_data = {method: real_results[target_task]['full'][method]["safe_task_completion_rate"] for method in method_list}  # Example: second metric is 90% of first
    
    ax = plot_real_task(real_results[target_task], plot_partial=False, plot_legend=True,
                        metric1_results=metric1_data, metric2_results=metric2_data,
                        metric1_key='Task Completion Rate', metric2_key='Safe Task Completion Rate')
    
    # Example 2: Using nested dict structure in results
    # real_results_nested = {
    #     'Shelve Item': {
    #         'full': {
    #             method: {
    #                 'metric1': real_results['Shelve Item']['full'][method],
    #                 'metric2': real_results['Shelve Item']['full'][method] * 0.9
    #             }
    #             for method in method_list
    #         }
    #     }
    # }
    # ax = plot_real_task(real_results_nested['Shelve Item'], plot_partial=False, plot_legend=True,
    #                     metric1_key='metric1', metric2_key='metric2')
    
    # ax.legend(loc='upper left', fontsize=12, frameon=False, ncol=1, columnspacing=1, bbox_to_anchor=(1.15, 0.8))
    plt.savefig(f'resources/figures/{target_task}.pdf', dpi=300, bbox_inches='tight')