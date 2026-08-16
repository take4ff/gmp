import os
import csv
import itertools
import math

def count_dinucleotide(codon, target):
    """
    コドン内の特定のジヌクレオチド（'CG'や'TA'）の個数をカウントします。
    """
    return codon.count(target)

def safe_log(val):
    """
    安全に自然対数(底 e)を計算します。
    値が0以下の場合、非常に小さな値(1e-9)で補正します。
    """
    if val <= 0:
        val = 1e-9
    return math.log(val)

def check_transition(codon1, codon2):
    """
    1塩基置換がトランジション（A <-> G, C <-> T）であるかどうかを判定します。
    """
    b1, b2 = None, None
    for i in range(3):
        if codon1[i] != codon2[i]:
            b1 = codon1[i]
            b2 = codon2[i]
            break
            
    if b1 is None or b2 is None:
        return 0
        
    transition_pairs = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}
    if (b1, b2) in transition_pairs:
        return 1
    return 0

def load_codon_usage(csv_path):
    """
    コドン使用頻度CSVを読み込み、データを辞書として返します。
    """
    data = {}
    if not os.path.exists(csv_path):
        print(f"エラー: ファイルが見つかりません - {csv_path}")
        return None
    
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            codon = row.get('Type')
            if codon:
                codon = codon.strip()
                data[codon] = {
                    'aa': row.get('aa', '').strip() if row.get('aa') else None,
                    'per_thousand': float(row.get('per thousand', 0.0)),
                    'RSCU': float(row.get('RSCU', 0.0))
                }
    return data

def calculate_relative_adaptability(codon_data):
    """
    各アミノ酸グループ内で、コドンの相対的適応度 w = f / f_max を計算します。
    """
    max_freq_by_aa = {}
    for codon, info in codon_data.items():
        aa = info['aa']
        freq = info['per_thousand']
        if aa:
            if aa not in max_freq_by_aa or freq > max_freq_by_aa[aa]:
                max_freq_by_aa[aa] = freq
                
    w_dict = {}
    for codon, info in codon_data.items():
        aa = info['aa']
        freq = info['per_thousand']
        if aa and aa in max_freq_by_aa and max_freq_by_aa[aa] > 0:
            w_dict[codon] = freq / max_freq_by_aa[aa]
        else:
            w_dict[codon] = 0.0
    return w_dict

def generate_mutation_pairs():
    bases = ['A', 'T', 'G', 'C']
    codons = [''.join(p) for p in itertools.product(bases, repeat=3)]
    mutation_pairs = []
    for codon in codons:
        for i in range(3):
            for b in bases:
                if codon[i] != b:
                    mutated_codon = codon[:i] + b + codon[i+1:]
                    mutation_pairs.append((codon, mutated_codon))
    return mutation_pairs

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # パス設定
    human_csv  = os.path.join(project_root, 'reference', 'codon', 'human_codonUsage.csv')
    scv2_csv   = os.path.join(project_root, 'reference', 'codon', 'SCV2_codonUsage.csv')
    output_csv = os.path.join(project_root, 'reference', 'codon', 'SCV2_host_adaptation_features.csv')
    
    # データのロード
    human_data = load_codon_usage(human_csv)
    scv2_data = load_codon_usage(scv2_csv)
    
    if not human_data or not scv2_data:
        print("エラー: データの読み込みに失敗しました。処理を中断します。")
        return
        
    # SCV2データにアミノ酸情報を補完する (human_dataからマッピング)
    for codon in scv2_data:
        if codon in human_data:
            scv2_data[codon]['aa'] = human_data[codon]['aa']
            
    # 相対適応度 w の算出 (w=1.0 がそのアミノ酸における最適コドン)
    w_human = calculate_relative_adaptability(human_data)
    w_scv2 = calculate_relative_adaptability(scv2_data)
    
    # 1塩基置換変異ペアの生成
    mutation_pairs = generate_mutation_pairs()
    
    output_rows = []
    for before, after in mutation_pairs:
        # アミノ酸情報の取得
        aa_before = human_data.get(before, {}).get('aa')
        aa_after = human_data.get(after, {}).get('aa')
        
        # 1. 同義変異判定 (is_synonymous)
        is_synonymous = 1 if (aa_before and aa_after and aa_before == aa_after) else 0
        
        # 2. 最適コドン変化の離散フラグ
        is_optimal_before = 1 if w_human.get(before, 0.0) == 1.0 else 0
        is_optimal_after = 1 if w_human.get(after, 0.0) == 1.0 else 0
        
        optimal_to_optimal = 1 if (is_optimal_before == 1 and is_optimal_after == 1) else 0
        non_optimal_to_optimal = 1 if (is_optimal_before == 0 and is_optimal_after == 1) else 0
        optimal_to_non_optimal = 1 if (is_optimal_before == 1 and is_optimal_after == 0) else 0
        
        # 3. 塩基置換タイプ（トランジション/トランスバージョン）と相互作用
        is_transition = check_transition(before, after)
        
        # 4. CpG / UpA ジヌクレオチドの変化量
        cpg_before = count_dinucleotide(before, 'CG')
        cpg_after = count_dinucleotide(after, 'CG')
        cpg_diff = cpg_after - cpg_before
        
        upa_before = count_dinucleotide(before, 'TA')
        upa_after = count_dinucleotide(after, 'TA')
        upa_diff = upa_after - upa_before
        
        # 5. RSCU の値と変化量 (およびトランジションとの相互作用)
        rscu_h_before = human_data.get(before, {}).get('RSCU', 0.0)
        rscu_h_after = human_data.get(after, {}).get('RSCU', 0.0)
        human_RSCU_diff = rscu_h_after - rscu_h_before
        
        rscu_v_before = scv2_data.get(before, {}).get('RSCU', 0.0)
        rscu_v_after = scv2_data.get(after, {}).get('RSCU', 0.0)
        SCV2_RSCU_diff = rscu_v_after - rscu_v_before
        
        transition_human_RSCU_diff = is_transition * human_RSCU_diff
        transition_SCV2_RSCU_diff = is_transition * SCV2_RSCU_diff
        
        # 6. コドン頻度の値と変化量 (dfreq)
        f_h_before = human_data.get(before, {}).get('per_thousand', 0.0)
        f_h_after = human_data.get(after, {}).get('per_thousand', 0.0)
        human_freq_diff = f_h_after - f_h_before
        
        f_v_before = scv2_data.get(before, {}).get('per_thousand', 0.0)
        f_v_after = scv2_data.get(after, {}).get('per_thousand', 0.0)
        SCV2_freq_diff = f_v_after - f_v_before
        
        # 7. 相対適応度 (CAIベース) の値と変化量
        w_h_before = w_human.get(before, 0.0)
        w_h_after = w_human.get(after, 0.0)
        human_cai_diff = w_h_after - w_h_before
        
        w_v_before = w_scv2.get(before, 0.0)
        w_v_after = w_scv2.get(after, 0.0)
        SCV2_cai_diff = w_v_after - w_v_before
        
        # 8. コドン頻度対数比の絶対値（距離）の値と減少量
        ratio_before = f_v_before / f_h_before if f_h_before > 0 else 1.0
        ratio_after = f_v_after / f_h_after if f_h_after > 0 else 1.0
        
        log_ratio_before = safe_log(ratio_before)
        log_ratio_after = safe_log(ratio_after)
        
        dist_log_before = abs(log_ratio_before)
        dist_log_after = abs(log_ratio_after)
        host_distance_log_ratio_diff = dist_log_before - dist_log_after
        
        # 9. RSCU対数比の絶対値（距離）の値と減少量
        rscu_ratio_before = rscu_v_before / rscu_h_before if rscu_h_before > 0 else 1.0
        rscu_ratio_after = rscu_v_after / rscu_h_after if rscu_h_after > 0 else 1.0
        
        log_rscu_before = safe_log(rscu_ratio_before)
        log_rscu_after = safe_log(rscu_ratio_after)
        
        dist_rscu_before = abs(log_rscu_before)
        dist_rscu_after = abs(log_rscu_after)
        host_distance_RSCU_ratio_diff = dist_rscu_before - dist_rscu_after
        
        output_rows.append({
            'before_codon': before,
            'after_codon': after,
            'is_synonymous': is_synonymous,
            'optimal_to_optimal': optimal_to_optimal,
            'non_optimal_to_optimal': non_optimal_to_optimal,
            'optimal_to_non_optimal': optimal_to_non_optimal,
            'is_transition': is_transition,
            'transition_human_RSCU_diff': transition_human_RSCU_diff,
            'transition_SCV2_RSCU_diff': transition_SCV2_RSCU_diff,
            'CpG_diff': cpg_diff,
            'UpA_diff': upa_diff,
            'human_RSCU_before': rscu_h_before,
            'human_RSCU_diff': human_RSCU_diff,
            'SCV2_RSCU_before': rscu_v_before,
            'SCV2_RSCU_diff': SCV2_RSCU_diff,
            'human_freq_before': f_h_before,
            'human_freq_diff': human_freq_diff,
            'SCV2_freq_before': f_v_before,
            'SCV2_freq_diff': SCV2_freq_diff,
            'human_CAI_before': w_h_before,
            'human_CAI_diff': human_cai_diff,
            'SCV2_CAI_before': w_v_before,
            'SCV2_CAI_diff': SCV2_cai_diff,
            'host_distance_log_ratio_before': dist_log_before,
            'host_distance_log_ratio_diff': host_distance_log_ratio_diff,
            'host_distance_RSCU_ratio_before': dist_rscu_before,
            'host_distance_RSCU_ratio_diff': host_distance_RSCU_ratio_diff
        })
        
    # 保存処理
    fieldnames = [
        'before_codon', 'after_codon', 'is_synonymous', 
        'optimal_to_optimal', 'non_optimal_to_optimal', 'optimal_to_non_optimal',
        'is_transition', 'transition_human_RSCU_diff', 'transition_SCV2_RSCU_diff',
        'CpG_diff', 'UpA_diff', 
        'human_RSCU_before', 'human_RSCU_diff',
        'SCV2_RSCU_before', 'SCV2_RSCU_diff',
        'human_freq_before', 'human_freq_diff',
        'SCV2_freq_before', 'SCV2_freq_diff',
        'human_CAI_before', 'human_CAI_diff',
        'SCV2_CAI_before', 'SCV2_CAI_diff',
        'host_distance_log_ratio_before', 'host_distance_log_ratio_diff',
        'host_distance_RSCU_ratio_before', 'host_distance_RSCU_ratio_diff'
    ]
    
    try:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"成功: '{output_csv}' を生成しました。 (行数: {len(output_rows)})")
    except Exception as e:
        print(f"エラー: ファイルの書き込みに失敗しました - {output_csv}. エラー: {e}")

if __name__ == '__main__':
    main()
