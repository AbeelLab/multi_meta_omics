import matplotlib.pyplot as plt
import numpy as np

def make_multi_bar_chart(x_ticks: list[str],
                         bar_width: float,
                         figsize: (float, float),
                         benchmarking_labels: list[str],
                         plotting_labels: list[str],
                         values: dict,
                         stds: dict,
                         colors: dict,
                         y_label: str,
                         x_label: str,
                         title: str,
                         y_lim: (float, float),
                         save_to: str,
                         alphas: dict=None,
                         dashed_line_mean: float=None,
                         dashed_line_error: float=None,
                         dashed_line_color: str=None,
                         last_bar_hatch: str=None,
                         label_rotation: float=0):

    if alphas is None:
        alphas = {x: 1 for x in benchmarking_labels}
        
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Plotting code adapted from: https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(x_ticks))

    width = bar_width
    multiplier = 0

    fig, ax = plt.subplots(figsize=figsize)

    if dashed_line_mean is not None:
        # Random guessing
        plt.axhline(y=0.5,
                    color="black",
                    linewidth=1.5)
        
        plt.axhline(y=dashed_line_mean,
                    color=dashed_line_color,
                    ls="dashed",
                    linewidth=1.5,
                    dashes=(3, 3))
        plt.fill_between(x,
                         dashed_line_mean - dashed_line_error,
                         dashed_line_mean + dashed_line_error,
                         color=dashed_line_color,
                         alpha=0.2,
                         linewidth=0)

    last_bars = len(benchmarking_labels) - 1
    for i, label in enumerate(benchmarking_labels):
        offset = width * multiplier
        score = values[label]
        
        error = stds[label]
        rects = ax.bar(x + offset, score, width,
                       label=plotting_labels[i],
                       color=colors[label],
                       edgecolor="black",
                       alpha = alphas[label],
                       yerr=error,
                       ls='none',
                       ecolor="black",
                       linewidth=1,
                       capsize=2)
        
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)

    tick_positions = x + (len(x_ticks) % 2) * (width / 2)
    tick_positions += int(len(benchmarking_labels) / 2) * width
    ax.set_xticks(tick_positions, x_ticks, fontsize="9")
    ax.legend(loc='upper left', ncols=len(x_ticks), fontsize="9")
    ax.set_title(title)

    ax.set_ylim(y_lim)

    ax.tick_params(axis='x', labelrotation=label_rotation)
    
    plt.savefig(save_to, dpi=300, bbox_inches="tight")
    plt.savefig(save_to.replace("png", "svg"), bbox_inches="tight")

    plt.close()
