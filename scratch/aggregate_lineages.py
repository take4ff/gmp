import pandas as pd
import os

def main():
    csv_path = "/mnt/ssd1/home3/aiba/gmp/meta_data/sequences-241017.csv"
    output_dir = "/mnt/ssd1/home3/aiba/gmp/scratch/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV from {csv_path} and counting lineages...")
    
    # メモリ節約のため、Pangolin列のみをチャンクで読み込む
    chunksize = 1000000
    counts = {}
    total_rows = 0
    
    for chunk in pd.read_csv(csv_path, usecols=['Pangolin'], chunksize=chunksize):
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")
        
        # 欠損値補完
        chunk['Pangolin'] = chunk['Pangolin'].fillna('Unknown')
        
        # カウント
        vc = chunk['Pangolin'].value_counts()
        for p, count in vc.items():
            counts[p] = counts.get(p, 0) + count
            
    print("Formatting counts...")
    df = pd.DataFrame(list(counts.items()), columns=['Pangolin', 'Count'])
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    
    # 割合の計算
    total_count = df['Count'].sum()
    df['Percentage'] = (df['Count'] / total_count) * 100
    
    # CSVに書き出し
    output_path = os.path.join(output_dir, "lineage_counts.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved all lineage counts to {output_path}")
    
    # 上位30件をプレビュー表示
    print("\n--- TOP 30 LINEAGES PREVIEW ---")
    print(df.head(30).to_string(index=False))

if __name__ == "__main__":
    main()
