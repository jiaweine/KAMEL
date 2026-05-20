import sys
sys.path.insert(0, '/root/KAMEL')
from pub_style import apply_style, add_minor_ticks, style_spines
apply_style()
import matplotlib.pyplot as plt
import pandas as pd

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
    "D": ["MLP", "SNN", "MLP-PLR", 'RealMLP', "ResNet", 'TabM'],
    "E": ["DCNv2", "DANets", "TabCaps"],
    "F": ["TabNet", "NODE", "GrowNet"],
    "G": ["TabPFN", "MNCA", "TabR", "MncaPFN-1-3000"],
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



def calculate_ranks_for_each_dataset(file_path):
    """
    Calculate ranks for each model across datasets, using mean scores and standard deviations.
    
    Parameters:
    file_path (str): Path to the Excel file with 'Mean' and 'Std' sheets.
    
    Returns:
    pd.DataFrame: A DataFrame with ranks calculated for each model across datasets.
    """
    # Load the data from Excel sheets
    mean_df = pd.read_excel(file_path, index_col=0)
    mean_df = mean_df.applymap(lambda x: -x if isinstance(x, float) else x)
    
    ranks_df = mean_df
    ranks_df = ranks_df.rank(axis=1, ascending=False)
    ranks_df = ranks_df.fillna(ranks_df.mean())
    print(ranks_df)
    return ranks_df


def plot_average_ranks(ranks_df, path):
    average_ranks = ranks_df.mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, len(average_ranks) * 0.28)))
    colors = []
    for model in average_ranks.index:
        col = '#AAAAAA'
        for key, vals in method_types.items():
            if model in vals:
                col = colors_type[key]; break
        colors.append(col)

    bars = ax.barh(average_ranks.index, average_ranks.values,
                   color=colors, edgecolor='white', lw=0.4, alpha=0.88)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.12, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}', va='center', fontsize=7, color='#333333')

    ax.set_xlim(average_ranks.min() - 1, average_ranks.max() + 2)
    ax.set_xlabel('Average Rank')
    ax.set_ylabel('Method')
    ax.grid(axis='x', ls='--', lw=0.4, alpha=0.3)
    ax.xaxis.set_minor_locator(__import__('matplotlib').ticker.AutoMinorLocator(2))
    style_spines(ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, dpi=600)
    plt.close(fig)


ranks_df = calculate_ranks_for_each_dataset('merged_result.xlsx')
ranks_df = ranks_df.rename(columns=names_dict)
plot_average_ranks(ranks_df, path='average_rank.pdf')