import pandas as pd
import matplotlib.pyplot as plt
import ast
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot mutation position distribution from evaluation CSV results.")
    parser.add_argument("--csv_path", type=str, default="outputs/transformer_260224/results/20260410_125114/test_predictions.csv", help="Path to test_predictions.csv")
    parser.add_argument("--out_path", type=str, default=None, help="Path to save the plot")
    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File not found {args.csv_path}")
        return

    # Load data
    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    # Function to parse string list like "[21846]" or "[3246, 4199]"
    def parse_list(x):
        try:
            return ast.literal_eval(x)
        except:
            return []

    df['target_position'] = df['target_position'].apply(parse_list)
    df['pred_position'] = df['pred_position'].apply(parse_list)

    # Exclude multi-target (co-occurrence) ground truth
    # "正解ラベルが共起のものは除外"
    df_single = df[df['target_position'].apply(len) == 1].copy()
    print(f"Filtered samples: {len(df_single)} / {len(df)}")

    # Extract the single integer from the list
    df_single['target_pos_val'] = df_single['target_position'].apply(lambda x: x[0])
    
    # For pred, take the first element if it exists (usually singleton if TOP_K_EVAL=1)
    df_single['pred_pos_val'] = df_single['pred_position'].apply(lambda x: x[0] if len(x) > 0 else -1)

    # Count frequencies
    target_counts = df_single['target_pos_val'].value_counts().sort_index()
    pred_counts = df_single['pred_pos_val'].value_counts().sort_index()

    # Create plot
    plt.figure(figsize=(15, 10))
    
    # Increase base font size
    plt.rcParams.update({'font.size': 16})

    plt.step(target_counts.index, target_counts.values, label='Ground Truth (Single Mutation)', color='blue', alpha=0.6, where='mid', linewidth=1.5)
    plt.step(pred_counts.index, pred_counts.values, label='Prediction (Top-1)', color='red', alpha=0.6, where='mid', linewidth=1.5)

    plt.title(f'Mutation Position Frequency: Ground Truth vs Prediction\n({os.path.basename(args.csv_path)}, Excl. Co-occurrence)', fontsize=22, fontweight='bold', pad=20)
    plt.xlabel('Genomic Position', fontsize=18, fontweight='bold')
    plt.ylabel('Frequency (Count)', fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, prop={'weight': 'bold'})
    plt.xticks(fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlim(0, 30000) 

    plt.tight_layout()

    # Save
    out_file = args.out_path if args.out_path else args.csv_path.replace('.csv', '_mutation_dist.png')
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to: {out_file}")

if __name__ == "__main__":
    main()
