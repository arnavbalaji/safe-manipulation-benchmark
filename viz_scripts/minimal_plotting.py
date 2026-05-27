import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
sns.set_style('darkgrid')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Nimbus Roman', 'DejaVu Serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # makes math look Times-like too

# w/o health feedback with health feedback all datahealth-based episode filteringdamage-based data fitlering

method_name = 'DataMIL'
color_maps = {
    'w/o health feedback': 'C0', # (0.5, 0.5, 0.5),
    # 'all data': (0.7, 0.7, 0.7),
    'with health feedback': 'C1',
    # 'SR': 'C5',
    'all data': 'C2',
    # 'BR': 'C1',
    'health-filtered episodes': 'C3',
    'health-filtered datapoints': 'C4',
    method_name: 'C5',
}

alpha_map = {
    'w/o health feedback': 0.7,
    'with health feedback': 0.7,
    # 'SR': 0.7,
    'all data': 0.7,
    'health-filtered episodes': 0.7,
    'health-filtered datapoints': 0.7,
    # 'BR': 0.7,
    method_name: 1,
}

method_list = ['w/o health feedback', 'with health feedback', 'all data', 'health-filtered episodes',  'health-filtered datapoints']
_real_results = {
    'shelve_cereal_box': {
        'full': {
                'w/o health feedback': {'task_completion_rate': 0.73, 'safe_task_completion_rate': 0.13},
                'with health feedback': {'task_completion_rate': 0.70, 'safe_task_completion_rate': 0.53},
                'all data': {'task_completion_rate': 0.76, 'safe_task_completion_rate': 0.26},
                'health-filtered episodes': {'task_completion_rate': 0.86, 'safe_task_completion_rate': 0.6},
                'health-filtered datapoints': {'task_completion_rate': 0.73, 'safe_task_completion_rate': 0.33}
            }
    },
    'lift_egg': {
        'full': {
                'w/o health feedback': {'task_completion_rate': 0.5, 'safe_task_completion_rate': 0.1},
                'with health feedback': {'task_completion_rate': 0.63, 'safe_task_completion_rate': 0.6},
                'all data': {'task_completion_rate': 0.66, 'safe_task_completion_rate': 0.33},
                'health-filtered episodes': {'task_completion_rate': 0.56, 'safe_task_completion_rate': 0.43},
                'health-filtered datapoints': {'task_completion_rate': 0.76, 'safe_task_completion_rate': 0.7},
            },
    },
    'add_firewood': {
        'full': {
                'w/o health feedback': {'task_completion_rate': 0.73, 'safe_task_completion_rate': 0.2},
                'with health feedback': {'task_completion_rate': 0.8, 'safe_task_completion_rate': 0.7},
                'all data': {'task_completion_rate': 0.75, 'safe_task_completion_rate': 0.43},
                'health-filtered episodes': {'task_completion_rate': 0.9, 'safe_task_completion_rate': 0.85},
                'health-filtered datapoints': {'task_completion_rate': 0.7, 'safe_task_completion_rate': 0.65},
            }
    },
    'pour_water': {
        'full': {
                'w/o health feedback': {'task_completion_rate': 0.6, 'safe_task_completion_rate': 0.1},
                'with health feedback': {'task_completion_rate': 0.5, 'safe_task_completion_rate': 0.42},
                'all data': {'task_completion_rate': 0.75, 'safe_task_completion_rate': 0.23},
                'health-filtered episodes': {'task_completion_rate': 0.70, 'safe_task_completion_rate': 0.6},
                'health-filtered datapoints': {'task_completion_rate': 0.55, 'safe_task_completion_rate': 0.45},
            }        
    },
    'wipe_countertop': {
        'full': {
                'w/o health feedback': {'task_completion_rate': 0.9, 'safe_task_completion_rate': 0.05},
                'with health feedback': {'task_completion_rate': 0.95, 'safe_task_completion_rate': 0.8},
                'all data': {'task_completion_rate': 1.0, 'safe_task_completion_rate': 0.35},
                'health-filtered episodes': {'task_completion_rate': 0.93, 'safe_task_completion_rate': 0.9},
                'health-filtered datapoints': {'task_completion_rate': 0.85, 'safe_task_completion_rate': 0.75},
            }        
    }
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
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Set white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Bar width for each individual bar
    bar_width = 0.3
    # Gap between the two bars of the same method (0 = touching)
    gap_within_method = 0.0
    # Gap between different methods
    gap_between_methods = 0.15
    
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
    ax.set_xticks([])  # No x-axis labels (methods are in legend)
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
        # Create custom legend with methods and metric patterns
        import matplotlib.patches as mpatches
        
        # Create legend entries for all methods
        method_handles = []
        method_labels = []
        for k in keys:
            base_color = color_maps[k]
            base_alpha = alpha_map[k]
            # Create a patch with the method's color (solid, no pattern for method identification)
            method_patch = mpatches.Patch(facecolor=base_color, edgecolor='black', 
                                          alpha=base_alpha, label=k)
            method_handles.append(method_patch)
            method_labels.append(k)
        
        # Create pattern legend entries for metrics (using white color to show pattern clearly)
        pattern1_patch = mpatches.Patch(facecolor='white', edgecolor='black', 
                                        hatch=hatch1, label=metric1_key)
        pattern2_patch = mpatches.Patch(facecolor='white', edgecolor='black', 
                                        hatch=hatch2, label=metric2_key)
        
        # Combine method handles with pattern handles
        all_handles = method_handles + [pattern1_patch, pattern2_patch]
        all_labels = method_labels + [metric1_key, metric2_key]
        
        plt.legend(all_handles, all_labels, loc='upper left', fontsize=12, 
                  frameon=False, ncol=min(len(all_labels), 7), columnspacing=1, 
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

    target_task = 'wipe_countertop'
    
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