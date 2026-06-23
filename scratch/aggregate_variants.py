import pandas as pd
import numpy as np
import re
import os

def main():
    csv_path = "/mnt/ssd1/home3/aiba/gmp/meta_data/sequences-241017.csv"
    output_dir = "/mnt/ssd1/home3/aiba/gmp/scratch/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading CSV from {csv_path}...")
    # メモリ節約のため、必要な列のみをチャンクで読み込む
    chunksize = 1000000
    counts = {}
    
    # 日付パターンの正規表現 (YYYY-MM-DD または YYYY-MM)
    date_pat = re.compile(r'^(\d{4}-\d{2})')
    
    total_rows = 0
    for chunk in pd.read_csv(csv_path, usecols=['Pangolin', 'Collection_Date'], chunksize=chunksize):
        total_rows += len(chunk)
        print(f"Processed {total_rows} rows...")
        
        # 欠損値補完
        chunk['Pangolin'] = chunk['Pangolin'].fillna('Unknown')
        chunk['Collection_Date'] = chunk['Collection_Date'].fillna('Unknown')
        
        # 月の抽出 (YYYY-MM)
        def get_month(x):
            m = date_pat.match(str(x))
            return m.group(1) if m else 'Unknown'
            
        chunk['Month'] = chunk['Collection_Date'].apply(get_month)
        
        # グループ化してカウント
        grp = chunk.groupby(['Month', 'Pangolin']).size().reset_index(name='count')
        
        for _, row in grp.iterrows():
            m = row['Month']
            p = row['Pangolin']
            c = row['count']
            if m not in counts:
                counts[m] = {}
            counts[m][p] = counts[m].get(p, 0) + c
            
    print("Aggregation finished. Formatting data...")
    
    # データをフラットなリストに変換
    flat_data = []
    for month, p_counts in counts.items():
        for p, count in p_counts.items():
            flat_data.append({'Month': month, 'Pangolin': p, 'Count': count})
            
    df = pd.DataFrame(flat_data)
    
    # 不明な月や極端に古い/未来のデータ、'Unknown'をフィルタリング (例: 2019年〜2026年)
    # 有効な年月フォーマット (YYYY-MM) に一致するものだけを残す
    df = df[df['Month'].str.match(r'^(201[9]|202[0-6])-\d{2}$')]
    
    # CSVに保存
    df.to_csv(os.path.join(output_dir, "monthly_variants_raw.csv"), index=False)
    print("Saved monthly_variants_raw.csv")
    
    # 1. 各月の上位5系統の表を作成
    monthly_top = []
    for month in sorted(df['Month'].unique()):
        m_df = df[df['Month'] == month].sort_values(by='Count', ascending=False)
        total_month = m_df['Count'].sum()
        top5 = m_df.head(5)
        
        top5_list = []
        for _, row in top5.iterrows():
            pct = (row['Count'] / total_month) * 100
            top5_list.append(f"{row['Pangolin']} ({row['Count']:,} / {pct:.1f}%)")
            
        # 不足分を埋める
        while len(top5_list) < 5:
            top5_list.append("-")
            
        monthly_top.append({
            'Month': month,
            'Total': total_month,
            'Top 1': top5_list[0],
            'Top 2': top5_list[1],
            'Top 3': top5_list[2],
            'Top 4': top5_list[3],
            'Top 5': top5_list[4]
        })
        
    df_top = pd.DataFrame(monthly_top)
    df_top.to_csv(os.path.join(output_dir, "monthly_top_variants.csv"), index=False)
    print("Saved monthly_top_variants.csv")
    
    # 2. 全体で検出数が多い上位15系統の月別検出数（ピボット）
    overall_top = df.groupby('Pangolin')['Count'].sum().reset_index()
    # 'Unknown' は除外して上位15系統を選ぶ
    overall_top = overall_top[overall_top['Pangolin'] != 'Unknown']
    top15_variants = overall_top.sort_values(by='Count', ascending=False).head(15)['Pangolin'].tolist()
    
    # ピボットテーブル作成
    df_pivot = df[df['Pangolin'].isin(top15_variants)].pivot(index='Month', columns='Pangolin', values='Count').fillna(0).astype(int)
    # 月の順にソート
    df_pivot = df_pivot.sort_index()
    df_pivot.to_csv(os.path.join(output_dir, "monthly_top15_pivot.csv"))
    print("Saved monthly_top15_pivot.csv")
    
    # マークダウン用に一部結果を表示
    print("\n--- MONTHLY TOP 3 VARIANTS PREVIEW ---")
    print(df_top[['Month', 'Total', 'Top 1', 'Top 2', 'Top 3']].to_string(index=False))

if __name__ == "__main__":
    main()
