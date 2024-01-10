import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import ticker as mplticker
import pickle
from misc import utils


def plot_aj(df: pd.DataFrame,
            title: str,
            estimate_col: str = "mean",
            lower_col: str = "lower",
            upper_col: str = "upper",
            t_col: str = "event_at",
            condition_col: str = "condition",
            y_axis_label_decimals: int = 0, ):
    """

    Makes a line plot with shaded error region.

    """
    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = df[condition_col].drop_duplicates().sort_values(ascending=False)
    # Switched from a built-in palette to specific colors to look fancier.
    # This will now break if there are > 2 conditions.
    palette = ["#FF9E4D", "#FF7400", "#38BCBC", "#009B9B"]

    i = 0
    for condition in conditions:
        condition_df = df[df[condition_col] == condition].sort_values(t_col)
        line_color = palette[i + 1]
        error_color = palette[i]
        i += 2
        sns.lineplot(
            data=condition_df,
            x=t_col,
            y=estimate_col,
            ax=ax,
            color=line_color,
            label=condition,
        )

        ax.fill_between(
            condition_df[t_col],
            condition_df[lower_col],
            condition_df[upper_col],
            facecolor=error_color,
            edgecolor=line_color,
            alpha=0.5,
            linewidth=1,
        )

    ax.set_xticks(range(0, int(df[t_col].max() + 1), 20))
    ax.set_xlabel("Days since COVID-19 Index")
    ax.set_ylabel("Cumulative Incidence")
    ax.yaxis.set_major_formatter(mplticker.PercentFormatter(1.0, decimals=y_axis_label_decimals))

    ax.legend(facecolor="white")
    ax.set_title(title)
    sns.set_theme()
    plt.show()




if __name__ == '__main__':
    infile = r'../data/recover/output/results/Paxlovid-n3c-all-narrow/1-any_pasc-cumIncidence-ajf1w-ajf0w.pkl'
    with open(infile, 'rb') as f:
        ajf1w, ajf0w = pickle.load(f)

    title = 'PASC'
    # ax = plt.subplot(111)
    fig, ax = plt.subplots(figsize=(8, 6))
    # ajf1.plot(ax=ax)
    palette = ["#FF9E4D", "#FF7400", "#38BCBC", "#009B9B"]

    ajf1w.plot(ax=ax, loc=slice(0., 180))  # 0, 180
    i = 0
    line_color = palette[i + 1]
    error_color = palette[i]

    sns.lineplot(
        data=ajf1w.cumulative_density_, #.loc[ajf1w.cumulative_density_.index < 180, :],
        x=ajf1w.cumulative_density_.index, #[ajf1w.cumulative_density_.index < 180],
        y='CIF_1',
        ax=ax,
        color=line_color,
        label='Paxlovid',
    )

    ax.fill_between(
        ajf1w.cumulative_density_['CIF_1'],
        ajf1w.confidence_interval_['Paxlovid_upper_0.95'],
        ajf1w.confidence_interval_['Paxlovid_lower_0.95'],
        facecolor=error_color,
        edgecolor=line_color,
        alpha=0.5,
        linewidth=1,
    )

    plt.xlim([0, 180])

    # ajf0.plot(ax=ax)
    # ajf0w.plot(ax=ax, loc=slice(0., 180))
    # add_at_risk_counts(ajf1w, ajf0w, ax=ax)

    # ax.set_xticks(range(0, int(df[t_col].max() + 1), 20))
    ax.set_xlabel("Days since COVID-19 Index")
    ax.set_ylabel("Cumulative Incidence")
    ax.yaxis.set_major_formatter(mplticker.PercentFormatter(1.0, decimals=0))

    ax.legend(facecolor="white")
    ax.set_title(title)
    sns.set_theme()
    plt.show()
