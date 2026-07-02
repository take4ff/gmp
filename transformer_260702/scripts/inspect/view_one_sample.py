# --- transformer_260702/scripts/view_one_sample.py ---
# DuckDB から 1 サンプルを取得し、特徴量を人間が読みやすい形で表示・CSV 保存する。
# Usage: python -m transformer_260702.scripts.view_one_sample
import os
import pickle
import pandas as pd
from transformer_260702 import config
from transformer_260702.db.connection import get_db_path, connect_db

BASE_VOCABS_INV  = {1: 'A', 2: 'T', 3: 'C', 4: 'G', 5: 'N', 6: 'n', 0: 'PAD'}
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
    db_path = get_db_path()
    print(f"[INFO] Connecting to database: {db_path}")
    con = connect_db(db_path, read_only=True)

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
        sample = con.execute("""
            SELECT s.sample_id, s.raw_path, s.path_length,
                   st.strain_name_ncbi, st.strain_name_usher,
                   st.clade, st.sample_count, st.strength_score_ncbi, st.strength_score_usher,
                   s.split_type, s.strength_score as sample_strength, s.collection_date
            FROM samples s
            JOIN strains st ON s.strain_id = st.strain_id
            LIMIT 1
        """).fetchone()

    (sample_id, raw_path, path_length, strain_name_ncbi, strain_name_usher,
     clade, sample_count, strength_score_ncbi, strength_score_usher,
     split_type, sample_strength, collection_date) = sample

    features = con.execute("""
        SELECT timestep, cooccurrence, cat_features, num_features
        FROM features
        WHERE sample_id = ?
        ORDER BY timestep
    """, [sample_id]).fetchall()
    con.close()

    cat_rows, num_rows = [], []
    split_labels = {0: 'Train', 1: 'Valid', 2: 'Test'}

    for timestep, cooc, cat_blob, num_blob in features:
        cat_feat = pickle.loads(cat_blob)
        num_feat = pickle.loads(num_blob)

        bef_base = BASE_VOCABS_INV.get(cat_feat[0], '?')
        base_pos  = cat_feat[1]
        aft_base  = BASE_VOCABS_INV.get(cat_feat[2], '?')
        codon_pos = cat_feat[3]
        bef_aa    = AA_VOCABS_INV.get(cat_feat[4], '?')
        aa_pos    = cat_feat[5]
        aft_aa    = AA_VOCABS_INV.get(cat_feat[6], '?')
        region    = PROTEIN_VOCABS_INV.get(cat_feat[7], '?')
        is_syn    = "Synonymous" if cat_feat[8] == 1 else "Non-Synonymous"

        ctx_w = getattr(config, 'CONTEXT_WINDOW', 3)
        n_ctx = 2 * ctx_w
        if len(cat_feat) >= 9 + n_ctx:
            ctx = [BASE_VOCABS_INV.get(cat_feat[9 + i], '?') for i in range(n_ctx)]
            left_context  = ''.join(ctx[:ctx_w])
            right_context = ''.join(ctx[ctx_w:])
            full_context  = f"{left_context}[{bef_base}->{aft_base}]{right_context}"
        else:
            left_context = right_context = full_context = '---'

        mut_str    = f"{bef_base}{base_pos}{aft_base}"
        aa_mut_str = f"{bef_aa}{aa_pos}{aft_aa}" if aa_pos > 0 or cat_feat[4] != 22 else "-"
        ts_label   = f"{timestep}" if cooc == 1 else f"{timestep} (cooc:{cooc})"

        cat_rows.append({'TS': ts_label, 'Mutation': mut_str, 'Full Context': full_context,
                         'CodonPos': codon_pos, 'AA Mutation': aa_mut_str, 'Region': region,
                         'Type': is_syn, 'Left Context': left_context, 'Right Context': right_context,
                         'Raw Cat Features': str(cat_feat)})
        num_rows.append({'TS': ts_label, 'Mutation': mut_str,
                         'Freq': f"{num_feat[0]:.1f}", 'Hydro': f"{num_feat[1]:.2f}",
                         'Charge': f"{num_feat[2]:.2f}", 'Size': f"{num_feat[3]:.2f}",
                         'BLSM62': f"{num_feat[4]:.1f}", 'PAM250': f"{num_feat[5]:.1f}",
                         'LogRatioDiff': f"{num_feat[6]:.4f}", 'LogRatioBef': f"{num_feat[7]:.4f}",
                         'RSCU_Human_Diff': f"{num_feat[8]:.4f}", 'RSCU_SCV2_Diff': f"{num_feat[9]:.4f}",
                         'Opt2Opt': f"{num_feat[10]:.0f}", 'NonOpt2Opt': f"{num_feat[11]:.0f}",
                         'Opt2NonOpt': f"{num_feat[12]:.0f}", 'IsTrans': f"{num_feat[13]:.0f}",
                         'TransHumanRSCU': f"{num_feat[14]:.4f}", 'TransSCV2RSCU': f"{num_feat[15]:.4f}",
                         'CpGDiff': f"{num_feat[16]:.0f}", 'UpADiff': f"{num_feat[17]:.0f}",
                         'HumanRSCUBef': f"{num_feat[18]:.4f}", 'SCV2RSCUBef': f"{num_feat[19]:.4f}",
                         'HumanFreqBef': f"{num_feat[20]:.2f}", 'HumanFreqDiff': f"{num_feat[21]:.2f}",
                         'SCV2FreqBef': f"{num_feat[22]:.2f}", 'SCV2FreqDiff': f"{num_feat[23]:.2f}",
                         'HumanCAIBef': f"{num_feat[24]:.4f}", 'HumanCAIDiff': f"{num_feat[25]:.4f}",
                         'SCV2CAIBef': f"{num_feat[26]:.4f}", 'SCV2CAIDiff': f"{num_feat[27]:.4f}",
                         'RSCURatioBef': f"{num_feat[28]:.4f}", 'RSCURatioDiff': f"{num_feat[29]:.4f}",
                         'CumSyn': f"{num_feat[30]:.0f}", 'CumNonSyn': f"{num_feat[31]:.0f}"})

    df_cat      = pd.DataFrame(cat_rows)
    df_num      = pd.DataFrame(num_rows)
    df_combined = pd.merge(df_cat, df_num, on=['TS', 'Mutation'])

    output_dir = os.path.join(config.OUTPUT_DIR, 'scripts', 'samples')
    os.makedirs(output_dir, exist_ok=True)
    info_csv_path     = os.path.join(output_dir, f"sample_{sample_id}_info.csv")
    combined_csv_path = os.path.join(output_dir, f"sample_{sample_id}.csv")

    info_data = {
        'Metric': ['Sample ID', 'Strain (NCBI Pango)', 'Strain (UShER)', 'Clade (Nextcl)',
                   'Collection Date', 'Split Type', 'Path Length',
                   'Strain Strength (NCBI)', 'Strain Strength (UShER)',
                   'Strain Sample Count', 'Sample Strength', 'Raw Path'],
        'Value': [sample_id, strain_name_ncbi, strain_name_usher,
                  clade or 'N/A', collection_date or 'N/A',
                  split_labels.get(split_type, 'Unknown'), path_length,
                  f"{strength_score_ncbi:.4f}", f"{strength_score_usher:.4f}",
                  sample_count, f"{sample_strength:.4f}", raw_path]
    }
    pd.DataFrame(info_data).to_csv(info_csv_path, index=False)
    df_combined.to_csv(combined_csv_path, index=False)

    print("=" * 100)
    print(f"SAMPLE INFORMATION")
    print(f"   Sample ID      : {sample_id}")
    print(f"   Strain (NCBI)  : {strain_name_ncbi}")
    print(f"   Strain (UShER) : {strain_name_usher}")
    print(f"   Clade (Nextcl) : {clade or 'N/A'}")
    print(f"   Collection Date: {collection_date or 'N/A'}")
    print(f"   Split Type     : {split_labels.get(split_type, 'Unknown')}")
    print(f"   Path Length    : {path_length} steps")
    print(f"   NCBI Strength  : {strength_score_ncbi:.4f}")
    print(f"   UShER Strength : {strength_score_usher:.4f} (sample count: {sample_count:,})")
    print(f"   Sample Strength: {sample_strength:.4f}")
    print(f"   Raw Path       : {raw_path}")
    print("=" * 100)
    print(f"\nCATEGORICAL FEATURES ({len(df_cat.columns) - 1} dimensions mapped)")
    print(df_cat.to_string(index=False))
    print("\nNUMERICAL FEATURES (10 Dimensions)")
    print(df_num.to_string(index=False))
    print("=" * 100)
    print(f"\nCSV FILES EXPORTED")
    print(f"   Info    : {info_csv_path}")
    print(f"   Combined: {combined_csv_path}")
    print("=" * 100)


if __name__ == '__main__':
    main()
