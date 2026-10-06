"""
Make nice plots with all_ir_results.csv
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns


def load_ir_results(filepath: str) -> pd.DataFrame:
    """
    Load the IR evaluation results from a CSV file.
    Columns expected:
    encoding, num_codebooks, nb_subspaces, codebook_size, 
    R@1, R@5, R@10, MRR@10, 
    R@1_min, R@5_min, R@10_min, MRR@10_min, 
    R@1_max, R@5_max, R@10_max, MRR@10_max
    """
    return pd.read_csv(filepath)


def plot_ir_results(df: pd.DataFrame, output_dir: str):
    """
    Create plots for IR results.
    Make three collumns, one for R@1, one for R@10 and one for MRR@10
    Always use the value of min+max/2 as the value for the line plot.

    First, plot the ir metrics depending on codebook size, then ir metric depending on num_codebooks (color is encoding method, line style is codebook size), 
    then ir metric depending on codebook_size (color is encoding method, line style is num_codebooks).
    """
    df = df.copy()
    df = df[df["encoding"].isin(["pq", "rq-kmeans", "rq-module"])]

    metrics = ["R@1", "R@10", "MRR@10"]
    
    # Calculate average of min and max for each metric
    for metric in metrics:
        df[f"{metric}_avg"] = (df[f"{metric}_worst"] + df[f"{metric}_best"]) / 2
    
    fig, axes = plt.subplots(2, len(metrics), figsize=(18, 12))
    for i, metric in enumerate(metrics):
        metric_col = f"{metric}_avg"
        # graphs 1,i and 2,i
        columns = ["codebook_size", "num_codebooks"]
        columns_names = ["V", "Docid Length"]
        titles = [f"{metric} vs V", f"{metric} vs Docid Length"]
        for j in range(2):
            x_col = columns[j]
            title = titles[j]
            ax = axes[j, i]
            
            # Convert x values to categorical (string) for proper spacing
            df_plot = df.copy()
            
            # Create display name columns for legend
            df_plot['Docid Length'] = df_plot['num_codebooks']
            df_plot['V'] = df_plot['codebook_size']

            # if j = 0, remove num_codebooks = 4
            if j == 0:
                df_plot = df_plot[df_plot["num_codebooks"] != 4]
            
            # Get unique x values for proper ordering
            x_order = sorted(df_plot[x_col].unique())
            x_order_str = [str(x) for x in x_order]
            
            # Create categorical column with explicit ordering
            df_plot[x_col + '_cat'] = pd.Categorical(
                df_plot[x_col].astype(str), 
                categories=x_order_str, 
                ordered=True
            )
            
            sns.lineplot(
                data=df_plot, 
                x=x_col + '_cat', 
                y=metric_col, 
                hue="encoding", 
                style="Docid Length" if j == 0 else "V", 
                markers=True,
                dashes=False,
                ax=ax
            )

            ax.set_title(title)
            ax.set_xlabel(columns_names[j])
            ax.set_ylabel(metric)
            ax.set_xticks(range(len(x_order_str)))
            ax.set_xticklabels(x_order_str)

            ax.grid(True, alpha=0.3)
            
            # Handle legend - only show on the first subplot
            if j == 0 and i == 2:
                ax.legend(title='(Encoding, Docid Length)', bbox_to_anchor=(1.05, 1), loc='upper left')
            elif j == 1 and i == 2:
                ax.legend(title='(Encoding, V)', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.get_legend().remove() if ax.get_legend() else None

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ir_results_plots.png"))
    print(f"Plots saved to {os.path.join(output_dir, 'ir_results_plots.png')}")
    plt.close()

def plot_pqrq_results(df: pd.DataFrame, output_dir: str):
    """
    Create plots for IR results, with encoding pq-rq
    Make three collumns, one for R@1, one for R@10 and one for MRR@10
    Always use the value of min+max/2 as the value for the line plot, and use the min-max range as the shaded area.

    V is constant in this, equal, to 256.
    Color is docid length, so num_codebooks x nb_subspaces
    First line is metric agains num_codebooks (L), with line style being nb_subspaces (C).
    Second line is metric against nb_subspaces (C), with line style being num_codebooks (L).
    """
    df = df.copy()
    df = df[df["encoding"].isin(["pq", "rq-kmeans", "pq-rq"])]
    df = df[df["codebook_size"] == 256]

    metrics = ["R@1", "R@10", "MRR@10"]
    
    # Calculate average of min and max for each metric
    for metric in metrics:
        df[f"{metric}_avg"] = (df[f"{metric}_worst"] + df[f"{metric}_best"]) / 2
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 6))
    for i, metric in enumerate(metrics):
        metric_col = f"{metric}_avg"
        # graphs 1,i and 2,i
        columns = ["num_codebooks", "nb_subspaces"]
        columns_names = ["L", "C"]
        titles = [f"{metric} vs L", f"{metric} vs C"]
        j = 0
        x_col = columns[j]
        title = titles[j]
        ax = axes[i]
        
        # Convert x values to categorical (string) for proper spacing
        df_plot = df.copy()
        
        # Create display name columns for legend
        df_plot['Docid Length'] = df_plot['num_codebooks'] * df_plot['nb_subspaces']
        df_plot['Docid Length'] = df_plot['Docid Length'].astype(int).astype(str)
        df_plot['L'] = df_plot['num_codebooks'].astype(int)
        df_plot['C'] = df_plot['nb_subspaces'].astype(int)
        # df_plot['V'] = df_plot['codebook_size'].astype(int)
        
        # Get unique x values for proper ordering
        x_order = sorted(df_plot[x_col].unique())
        x_order_str = [str(int(x)) for x in x_order]
        
        # Create categorical column with explicit ordering
        df_plot[x_col + '_cat'] = pd.Categorical(
            df_plot[x_col].astype(int).astype(str), 
            categories=x_order_str, 
            ordered=True
        )
        
        sns.lineplot(
            data=df_plot, 
            x=x_col + '_cat', 
            y=metric_col, 
            hue="Docid Length", 
            # style="Docid Length",
            markers=True,
            dashes=False,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel(columns_names[j])
        ax.set_ylabel(metric)
        ax.set_xticks(range(len(x_order_str)))
        ax.set_xticklabels(x_order_str)

        ax.grid(True, alpha=0.3)
        
        # Handle legend - only show on the first subplot
        if i == 2:
            ax.legend(title='Docid Length', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.get_legend().remove() if ax.get_legend() else None

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "ir_results_pqrq_plots.png"))
    print(f"Plots saved to {os.path.join(output_dir, 'ir_results_pqrq_plots.png')}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot IR evaluation results.")
    parser.add_argument("--results_csv", type=str, required=True, help="Path to the CSV file containing IR results.")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save the plots.")
    args = parser.parse_args()

    df = load_ir_results("src/data/data_prep/analysis/all_ir_results.csv")
    plot_ir_results(df, args.output_dir)
    df = load_ir_results("src/data/data_prep/analysis/all_ir_results_CLV.csv")
    plot_pqrq_results(df, args.output_dir)