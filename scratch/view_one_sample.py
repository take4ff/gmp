# --- scratch/view_one_sample.py ---
import duckdb
import pickle
import pandas as pd
import os

# ボキャブラリー逆引き用（表示を分かりやすくするため）
BASE_VOCABS_INV = {1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'N', 6: 'n', 0: 'PAD'}
AA_VOCABS_INV = {
    1: 'A', 2: 'R', 3: 'N', 4: 'D', 5: 'C', 6: 'Q', 7: 'E', 8: 'G',
    9: 'H', 10: 'I', 11: 'L', 12: 'K', 13: 'M', 14: 'F', 15: 'P', 16: 'S',
    17: 'T', 18: 'W', 19: 'Y', 20: 'V', 21: '*', 22: 'n', 0: 'PAD'
}
PROTEIN_VOCABS_INV = {
    1: 'non_coding1', 2: 'nsp1', 3: 'nsp2', 4: 'nsp3', 5: 'nsp4',
    6: 'nsp5', 7: 'nsp6', 8: 'nsp7', 9: 'nsp8', 10: 'nsp9',
    11: 'nsp10', 12: 'nsp12', 13: 'nsp13', 14: 'nsp14',
    15: 'nsp15', 16: 'nsp16', 17: 'non_coding2', 18: 'S', 19: 'non_coding3',
    20: 'ORF3a', 21: 'non_coding4', 22: 'E', 23: 'non_coding5', 24: 'M',
    25: 'non_coding6', 26: 'ORF6', 27: 'non_coding7', 28: 'ORF7a', 29: 'ORF7b',
    30: 'non_coding8', 31: 'ORF8', 32: 'non_coding9', 33: 'N', 34: 'non_coding10',
    35: 'ORF10', 36: 'non_coding11', 0: 'PAD'
}

def main():
    # 接続するDBパスを自動検出
    db_path = 'db/features_7dba6a01.duckdb'
    try:
        from transformer_260528.db.connection import get_db_path
        config_db_path = get_db_path()
        if os.path.exists(config_db_path):
            db_path = config_db_path
        else:
            # 存在しない場合は、dbフォルダ内のduckdbファイルを探索
            if os.path.exists('db'):
                db_files = [f for f in os.listdir('db') if f.endswith('.duckdb')]
                if db_files:
                    # 更新日時が新しい順にソートして最新のものを使用
                    db_files.sort(key=lambda x: os.path.getmtime(os.path.join('db', x)), reverse=True)
                    db_path = os.path.join('db', db_files[0])
    except Exception:
        pass

    print(f"[INFO] Connecting to database: {db_path}")
    con = duckdb.connect(db_path)
    
    # サンプルのメタデータを網羅的に取得
    sample = con.execute("""
        SELECT s.sample_id, s.raw_path, s.path_length, 
               st.strain_name_ncbi, st.strain_name_usher,
               st.clade, st.sample_count, st.strength_score_ncbi, st.strength_score_usher,
               s.split_type, s.strength_score as sample_strength, s.collection_date
        FROM samples s
        JOIN strains st ON s.strain_id = st.strain_id
        WHERE s.path_length BETWEEN 30 AND 40
        LIMIT 1
    """).fetchone()
    
    if not sample:
        # 見つからない場合は適当に1件
        sample = con.execute("""
            SELECT s.sample_id, s.raw_path, s.path_length, 
                   st.strain_name_ncbi, st.strain_name_usher,
                   st.clade, st.sample_count, st.strength_score_ncbi, st.strength_score_usher,
                   s.split_type, s.strength_score as sample_strength, s.collection_date
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
            LIMIT 1
        """).fetchone()
        
    sample_id, raw_path, path_length, strain_name_ncbi, strain_name_usher, clade, sample_count, strength_score_ncbi, strength_score_usher, split_type, sample_strength, collection_date = sample
    
    # 該当サンプルの特徴量を取得して timestep 昇順でソート
    features = con.execute("""
        SELECT timestep, cooccurrence, cat_features, num_features
        FROM features
        WHERE sample_id = ?
        ORDER BY timestep
    """, [sample_id]).fetchall()
    
    cat_rows = []
    num_rows = []
    
    split_labels = {0: 'Train', 1: 'Valid', 2: 'Test'}
    
    for timestep, cooc, cat_blob, num_blob in features:
        cat_feat = pickle.loads(cat_blob)
        num_feat = pickle.loads(num_blob)
        
        # カテゴリ特徴量のデコード
        bef_base = BASE_VOCABS_INV.get(cat_feat[0], '?')
        base_pos = cat_feat[1]
        aft_base = BASE_VOCABS_INV.get(cat_feat[2], '?')
        codon_pos = cat_feat[3]
        bef_aa = AA_VOCABS_INV.get(cat_feat[4], '?')
        aa_pos = cat_feat[5]
        aft_aa = AA_VOCABS_INV.get(cat_feat[6], '?')
        region = PROTEIN_VOCABS_INV.get(cat_feat[7], '?')
        is_syn = "Synonymous" if cat_feat[8] == 1 else "Non-Synonymous"
        
        # 隣の塩基 (15次元特徴量の場合のみデコード)
        if len(cat_feat) >= 15:
            left3 = BASE_VOCABS_INV.get(cat_feat[9], '?')
            left2 = BASE_VOCABS_INV.get(cat_feat[10], '?')
            left1 = BASE_VOCABS_INV.get(cat_feat[11], '?')
            right1 = BASE_VOCABS_INV.get(cat_feat[12], '?')
            right2 = BASE_VOCABS_INV.get(cat_feat[13], '?')
            right3 = BASE_VOCABS_INV.get(cat_feat[14], '?')
            
            left_context = f"{left3}{left2}{left1}"
            right_context = f"{right1}{right2}{right3}"
            full_context = f"{left_context}[{bef_base}->{aft_base}]{right_context}"
        else:
            left_context = "---"
            right_context = "---"
            full_context = "---"
            
        mut_str = f"{bef_base}{base_pos}{aft_base}"
        aa_mut_str = f"{bef_aa}{aa_pos}{aft_aa}" if cat_feat[5] > 0 or cat_feat[4] != 22 else "-"
        
        ts_label = f"{timestep}" if cooc == 1 else f"{timestep} (cooc:{cooc})"
        
        cat_rows.append({
            'TS': ts_label,
            'Mutation': mut_str,
            'Full Context': full_context,
            'CodonPos': codon_pos,
            'AA Mutation': aa_mut_str,
            'Region': region,
            'Type': is_syn,
            'Left Context': left_context,
            'Right Context': right_context,
            'Raw Cat Features': str(cat_feat)
        })
        
        # 数値特徴量のデコード
        num_rows.append({
            'TS': ts_label,
            'Mutation': mut_str,
            'Freq': f"{num_feat[0]:.1f}",
            'Hydro': f"{num_feat[1]:.2f}",
            'Charge': f"{num_feat[2]:.2f}",
            'Size': f"{num_feat[3]:.2f}",
            'BLSM62': f"{num_feat[4]:.1f}",
            'PAM250': f"{num_feat[5]:.1f}",
            'LogRatioDiff': f"{num_feat[6]:.4f}",
            'LogRatioBef': f"{num_feat[7]:.4f}",
            'RSCU_Human': f"{num_feat[8]:.4f}",
            'RSCU_SCV2': f"{num_feat[9]:.4f}"
        })
        
    con.close()
    
    df_cat = pd.DataFrame(cat_rows)
    df_num = pd.DataFrame(num_rows)
    
    # 2つの特徴量をマージした総合テーブルを作成
    # （結合キーに 'Full Context' などを加えるとマージエラーを避けるために TS, Mutation でマージ）
    df_combined = pd.merge(df_cat, df_num, on=['TS', 'Mutation'])
    
    # scratch ディレクトリにCSVとして保存
    os.makedirs('scratch/outputs', exist_ok=True)
    cat_csv_path = f"scratch/outputs/sample_{sample_id}_categorical.csv"
    num_csv_path = f"scratch/outputs/sample_{sample_id}_numerical.csv"
    combined_csv_path = f"scratch/outputs/sample_{sample_id}.csv"
    info_csv_path = f"scratch/outputs/sample_{sample_id}_info.csv"
    
    # メタデータ情報のCSV用DataFrame作成
    info_data = {
        'Metric': [
            'Sample ID',
            'Strain (NCBI Pango)',
            'Strain (UShER)',
            'Clade (Nextcl)',
            'Collection Date',
            'Split Type',
            'Path Length',
            'Strain Strength (NCBI)',
            'Strain Strength (UShER)',
            'Strain Sample Count',
            'Sample Strength',
            'Raw Path'
        ],
        'Value': [
            sample_id,
            strain_name_ncbi,
            strain_name_usher,
            clade if clade else 'N/A',
            collection_date if collection_date else 'N/A',
            split_labels.get(split_type, 'Unknown'),
            path_length,
            f"{strength_score_ncbi:.4f}",
            f"{strength_score_usher:.4f}",
            sample_count,
            f"{sample_strength:.4f}",
            raw_path
        ]
    }
    df_info = pd.DataFrame(info_data)
    
    # 各CSVを保存
    df_info.to_csv(info_csv_path, index=False)
    # df_cat.to_csv(cat_csv_path, index=False)
    # df_num.to_csv(num_csv_path, index=False)
    df_combined.to_csv(combined_csv_path, index=False)
    
    # 出力
    print("=" * 100)
    print(f"🧬 SAMPLE INFORMATION")
    print(f"   - Sample ID      : {sample_id}")
    print(f"   - Strain (NCBI)  : {strain_name_ncbi}")
    print(f"   - Strain (UShER) : {strain_name_usher}")
    print(f"   - Clade (Nextcl) : {clade if clade else 'N/A'}")
    print(f"   - Collection Date: {collection_date if collection_date else 'N/A'}")
    print(f"   - Split Type     : {split_labels.get(split_type, 'Unknown')}")
    print(f"   - Path Length    : {path_length} steps")
    print(f"   - NCBI Strength  : {strength_score_ncbi:.4f}")
    print(f"   - UShER Strength : {strength_score_usher:.4f} (sample count: {sample_count:,})")
    print(f"   - Sample Strength: {sample_strength:.4f}")
    print(f"   - Raw Path       : {raw_path}")
    print("=" * 100)
    
    print(f"\n📊 1. CATEGORICAL FEATURES ({len(df_cat.columns) - 1} dimensions mapped)")
    print(df_cat.to_string(index=False, justify='center'))
    
    print("\n📈 2. NUMERICAL FEATURES (10 Dimensions)")
    print(df_num.to_string(index=False, justify='center'))
    print("=" * 100)
    
    print("\n💾 3. CSV FILES EXPORTED")
    print(f"   - Sample Information CSV   : {info_csv_path}")
    print(f"   - Categorical Features CSV : {cat_csv_path} (Optional)")
    print(f"   - Numerical Features CSV   : {num_csv_path} (Optional)")
    print(f"   - Combined Flat Features CSV: {combined_csv_path}")
    print("=" * 100)

if __name__ == '__main__':
    main()
