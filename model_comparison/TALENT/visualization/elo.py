import sys
sys.path.insert(0, '/root/KAMEL')
from pub_style import apply_style, style_spines
apply_style()
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import random
import numpy as np

random.seed(42)
np.random.seed(42)

names_dict = {
    "dummy": "Dummy",
    "LogReg": "LR",
    "LinearRegression": "LR",
    "NCM": 'NCM',
    "NaiveBayes": "NB",
    "knn": "KNN",
    "svm": "SVM",
    "xgboost": "XGB",
    "catboost": "CatB",
    "RandomForest": "RForest",
    "lightgbm": "LightG",
    "tabpfn": "TabPFN",
    "mlp": "MLP",
    "resnet": "ResNet",
    "node": "NODE",
    "switchtab": "SwitchT",
    "tabnet": "TabNet",
    "tabcaps": "TabCaps",
    "tangos": "TANGOS",
    "danets": "DANets",
    "ftt": "FT-T",
    "autoint": "AutoInt",
    "dcn2": "DCNv2",
    "snn": "SNN",
    "tabtransformer": "TabT",
    "ptarl": "PTaRL",
    "grownet": "GrowNet",
    "tabr": "TabR",
    "dnnr": "DNNR",
    "realmlp": "RealMLP",
    "mlp_plr": "MLP-PLR",
    "excelformer": "ExcelF",
    "modernNCA": "MNCA",
    "tabm": "TabM",
}

method_types={
    "A": ["Dummy"],
    "B": ["LR", "NCM", "NB", "KNN", "SVM", "DNNR"],
    "C": ["XGB", "CatB", "RForest", "LightG"],
    "D": ["MLP", "SNN", "MLP-PLR", 'RealMLP', "ResNet"],
    "E": ["DCNv2", "DANets", "TabCaps"],
    "F": ["TabNet", "NODE", "GrowNet"],
    "G": ["TabPFN", "MNCA", "TabR"],
    "H": ["FT-T", "AutoInt", "ExcelF", "TabT"],
    "I": ["SwitchT", "TANGOS", "PTaRL"]
}

colors_type = {
    "A": "#DDDDDD",
    "B": "#4477AA",
    "C": "#228833",
    "D": "#CC6677",
    "E": "#5C6BC0",
    "F": "#009E73",
    "G": "#AA3377",
    "H": "#0072B2",
    "I": "#E69F00",
}



def calculate_elo_scores_with_missing_values(df, k_factor=32, initial_elo=1500):
    """
    Calculate ELO scores for multiple methods based on their performance across datasets,
    skipping rows with missing values for any method pair.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.

    Returns:
    - elo_scores: A dictionary of final ELO scores for each method.
    """
    methods = df.columns
    elo_scores = {method: initial_elo for method in methods}
    
    for _, row in df.iterrows():
        # Skip rows with missing values
        if row.isnull().any():
            continue
        performances = row
        
        # Compare each pair of methods
        for method_a in methods:
            for method_b in methods:
                if method_a == method_b:
                    continue

                rating_a = elo_scores[method_a]
                rating_b = elo_scores[method_b]
                performance_a = performances[method_a]
                performance_b = performances[method_b]

                # Determine the match result
                if performance_a > performance_b:
                    result_a, result_b = 1, 0
                elif performance_a < performance_b:
                    result_a, result_b = 0, 1
                else:
                    result_a, result_b = 0.5, 0.5

                # Calculate expected scores
                expected_a = 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
                expected_b = 1 / (1 + 10 ** ((rating_a - rating_b) / 400))

                # Update ELO scores
                elo_scores[method_a] += k_factor * (result_a - expected_a)
                elo_scores[method_b] += k_factor * (result_b - expected_b)

    return elo_scores


def calculate_average_elo_scores(df, k_factor=32, initial_elo=1500, num_shuffles=30):
    """
    Calculate average ELO scores for multiple methods across multiple randomized runs.

    Parameters:
    - df: DataFrame where each row is a dataset and each column is a method's performance.
    - k_factor: K-factor for the ELO calculation.
    - initial_elo: Initial ELO score for each method.
    - num_shuffles: Number of times to shuffle the datasets and calculate ELO.

    Returns:
    - avg_elo_scores: A dictionary of average ELO scores for each method across all shuffles.
    """
    methods = df.columns
    accumulated_elo_scores = {method: 0 for method in methods}

    for _ in range(num_shuffles):
        # Shuffle the datasets (rows)
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        # Calculate ELO scores for this shuffled order
        elo_scores = calculate_elo_scores_with_missing_values(shuffled_df, k_factor, initial_elo)
        
        # Accumulate ELO scores from this run
        for method in methods:
            accumulated_elo_scores[method] += elo_scores[method]

    # Calculate the average ELO scores
    avg_elo_scores = {method: score / num_shuffles for method, score in accumulated_elo_scores.items()}
    return avg_elo_scores


def plot_elo_scores(elo_scores, path='pics/elo_scores.pdf'):
    sorted_elo = dict(sorted(elo_scores.items(), key=lambda x: x[1], reverse=True))
    methods = list(sorted_elo.keys())
    scores  = list(sorted_elo.values())

    colors = []
    for model in methods:
        col = '#AAAAAA'
        for key, vals in method_types.items():
            if model in vals:
                col = colors_type[key]; break
        colors.append(col)

    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(methods) * 0.28)))
    ax.barh(methods, scores, color=colors, edgecolor='white', lw=0.4, alpha=0.88)
    ax.set_xlim(min(scores) - 50, max(scores) + 120)
    ax.set_xlabel('ELO Score')
    ax.set_ylabel('Method')
    ax.invert_yaxis()
    ax.grid(axis='x', ls='--', lw=0.4, alpha=0.3)
    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator(2))
    for i, v in enumerate(scores):
        ax.text(v + 8, i, f'{v:.1f}', va='center', fontsize=7, color='#333333')
    style_spines(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=600)
    plt.close(fig)


# Example usage
# Assume `df` is your DataFrame where rows are datasets and columns are methods.
# Example DataFrame to demonstrate
df_bin = pd.read_excel('acc_bin.xlsx')
df_bin = df_bin.rename(columns=names_dict)
df_bin = df_bin[df_bin.columns[1:]]
df_multi = pd.read_excel('acc_multi.xlsx')
df_multi = df_multi.rename(columns=names_dict)
df_multi = df_multi[df_multi.columns[1:]]
df_reg = pd.read_excel('rmse.xlsx')
df_reg = df_reg.rename(columns=names_dict)
df_reg = df_reg[df_reg.columns[1:]]
df_reg = df_reg.applymap(lambda x: -x if isinstance(x, float) else x)

df_all = pd.read_excel('merged_result.xlsx')
df_all = df_all.rename(columns=names_dict)
df_all = df_all[df_all.columns[1:]]

# Calculate ELO scores
elo_scores = calculate_average_elo_scores(df_bin)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_bin.pdf')

elo_scores = calculate_average_elo_scores(df_multi)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_multi.pdf')

elo_scores = calculate_average_elo_scores(df_reg)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_reg.pdf')

elo_scores = calculate_average_elo_scores(df_all)
print("Final ELO Scores:", elo_scores)
plot_elo_scores(elo_scores, path='elo_all.pdf')