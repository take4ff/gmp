"""共起変異の特徴量生成の順序非依存版（FEATURE_ORDER_INDEPENDENT=True）の検証。

従来(False): ステップ内の変異を位置昇順に逐次適用 → 同一コドン内の後の変異は「存在しない中間アミノ酸」に基づく。
新(True): 全変異をステップ開始時の状態に対して独立に評価し、同一コドン内は合成後のコドンを共有する。
検証: ①OFFで従来と同一（DB名・キャッシュのハッシュも不変）②相互作用の無いステップでは従来と同一の値
③記載順を入れ替えても各変異の特徴量・最終状態が同一 ④同一コドンのコドン単位特徴量が揃う ⑤文脈はステップ開始時
⑥最終状態は全変異適用後 ⑦状態の復元 ⑧前処理のラベルへの反映。
"""
import itertools
import random

import pytest

from transformer import config
from transformer.db import feature as F

AAS = sorted(set(F.DNA2Protein.values()))
CODONS = sorted(c for c, a in F.DNA2Protein.items() if a != '*')


def make_state(genome):
    n = len(genome)
    assert n % 3 == 0
    cods = [genome[i // 3 * 3: i // 3 * 3 + 3] for i in range(n)]
    return {'base': list(genome), 'protein': ['nsp1'] * n, 'aa_pos': [i // 3 + 1 for i in range(n)],
            'codon': cods, 'codon_pos': [i % 3 + 1 for i in range(n)]}


def _tables():
    rnd = random.Random(0)
    dissim = {(a, b): {'hydro': rnd.random(), 'charge': rnd.random(), 'size': rnd.random(), 'blsm': rnd.random()}
              for a in AAS for b in AAS}
    pam = {(a, b): float(rnd.randint(-8, 8)) for a in AAS for b in AAS}
    host = {(c1, c2): tuple(rnd.random() for _ in F._ADAPT_COLS) for c1 in CODONS for c2 in CODONS}
    return dissim, pam, host


DISSIM, PAM, HOST = _tables()


def freq_for(n):
    rnd = random.Random(1)
    return {f'{a}->{b}': [rnd.random() for _ in range(n)] for a in 'ACGT' for b in 'ACGT' if a != b}


def run_step(genome, step, ordered=True, cum=(0, 0)):
    """1ステップ分を Mutation_features_fast で生成。(特徴量リスト, 適用後の状態) を返す。"""
    state = make_state(genome)
    feats = F.Mutation_features_fast(step, state, freq_for(len(genome)), DISSIM, PAM, HOST, None, *cum)
    return feats, state


def by_mutation(step, feats):
    return {m: f for m, f in zip(step.split(','), feats)}


def random_genome(rnd, n_codons=24):
    return "".join(rnd.choice(CODONS) for _ in range(n_codons))


def random_step(rnd, genome, k, wrong_ref=0.1):
    pos = rnd.sample(range(1, len(genome) + 1), k)
    muts = []
    for p in pos:
        ref = genome[p - 1]
        bef = ref if rnd.random() > wrong_ref else rnd.choice([b for b in 'ACGT' if b != ref])
        muts.append(f"{bef}{p}{rnd.choice([b for b in 'ACGT' if b != bef])}")
    return muts


# ---------- ① OFF同値 ----------
def test_default_is_off_and_hashes_are_unchanged_for_the_existing_db(cfg):
    from transformer.db.connection import get_db_path
    from transformer.utils.io import get_config_hash
    assert config.FEATURE_ORDER_INDEPENDENT is False
    # 現行DB・株キャッシュのキー。特徴量設定を意図して変えた場合のみ更新すること（DB再構築が必要になる）
    assert get_db_path().endswith('features_b516bca4.duckdb')
    assert get_config_hash()[:8] == 'd9434d3c'


def test_flag_changes_db_name_and_cache_key(cfg):
    from transformer.db.connection import get_db_path
    from transformer.utils.io import get_config_hash
    off = (get_db_path(), get_config_hash())
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    on = (get_db_path(), get_config_hash())
    assert on[0] != off[0] and on[1] != off[1]


# ---------- ② 相互作用の無いステップでは従来と同一 ----------
def test_non_interacting_steps_equal_the_legacy_sequential_result(cfg):
    rnd = random.Random(2)
    checked = 0
    while checked < 300:
        genome = random_genome(rnd)
        muts = random_step(rnd, genome, rnd.randint(1, 4))
        pos = [int(m[1:-1]) for m in muts]
        if any(abs(a - b) <= 5 for a, b in itertools.combinations(pos, 2)):
            continue                                    # 同一コドン・文脈が相互作用しうる配置は除外
        step = ",".join(muts)
        cum = (rnd.randint(0, 5), rnd.randint(0, 5))
        cfg(FEATURE_ORDER_INDEPENDENT=False)
        old, old_state = run_step(genome, step, cum=cum)
        cfg(FEATURE_ORDER_INDEPENDENT=True)
        new, new_state = run_step(genome, step, cum=cum)
        assert new == old, step
        assert new_state == old_state                   # 最終状態も同一
        checked += 1


# ---------- ③ 記載順に依らない ----------
def test_features_and_final_state_do_not_depend_on_listed_order(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    rnd = random.Random(3)
    n_interacting = 0
    for _ in range(300):
        genome = random_genome(rnd, 12)                 # 短いゲノムで相互作用（同一コドン・5bp以内）を頻発させる
        muts = random_step(rnd, genome, rnd.randint(2, 4))
        ref_feats, ref_state = run_step(genome, ",".join(muts))
        ref = by_mutation(",".join(muts), ref_feats)
        pos = [int(m[1:-1]) for m in muts]
        n_interacting += any(abs(a - b) <= 5 for a, b in itertools.combinations(pos, 2))
        for perm in itertools.islice(itertools.permutations(muts), 6):
            step = ",".join(perm)
            feats, state = run_step(genome, step)
            assert by_mutation(step, feats) == ref, (muts, perm)
            assert state == ref_state
    assert n_interacting > 100                          # 相互作用するケースを十分に含んでいる


def test_legacy_sequential_result_does_depend_on_order_for_the_same_codon(cfg):
    """従来(False)は順序に依存する（Bで解消される問題の確認。現状仕様の記録）。"""
    cfg(FEATURE_ORDER_INDEPENDENT=False)
    genome = 'ATGGCTTTACCC'
    a, _ = run_step(genome, 'A1T,G3C')
    b, _ = run_step(genome, 'G3C,A1T')
    assert by_mutation('A1T,G3C', a) != by_mutation('G3C,A1T', b)


# ---------- ④ 同一コドン内でコドン単位の特徴量が揃う ----------
def test_same_codon_members_share_codon_level_features_and_keep_mutation_level_ones(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = 'ATGGCTTTACCC'                              # ATG(Met) → A1T,G3C で TTC(Phe)
    feats, _ = run_step(genome, 'A1T,G3C')
    (c1, n1), (c2, n2) = feats
    # コドン単位: aa_before/aa_after/同義判定、置換スコア(num[1:6])、ホスト適応(num[6:6+24]) が同一
    M, Phe = config.AA_VOCABS['M'], config.AA_VOCABS['F']
    assert c1[4] == c2[4] == M and c1[6] == c2[6] == Phe and c1[8] == c2[8] == 0
    assert n1[1:6] == n2[1:6] == [DISSIM[('M', 'F')][k] for k in ('hydro', 'charge', 'size', 'blsm')] + [PAM[('M', 'F')]]
    assert n1[6:6 + len(F._ADAPT_COLS)] == n2[6:6 + len(F._ADAPT_COLS)] == list(HOST[('ATG', 'TTC')])
    # 変異単位: 位置・変異前後塩基・codon_pos・再発頻度は各変異固有
    assert (c1[1], c2[1]) == (1, 3) and (c1[0], c1[2]) == (config.BASE_VOCABS['A'], config.BASE_VOCABS['T'])
    assert (c1[3], c2[3]) == (1, 3)
    assert n1[0] != n2[0]


def test_two_step_substitution_to_a_synonymous_codon_is_synonymous_for_both_members(cfg):
    """AGT(Ser) → A1T,G2C → TCT(Ser): 合成効果は同義。従来は中間のCysを経て両方が非同義、新は両方が同義。"""
    genome = 'AGTGCTTTACCC'
    cfg(FEATURE_ORDER_INDEPENDENT=False)
    legacy, _ = run_step(genome, 'A1T,G2C')
    assert [c[8] for c, _ in legacy] == [0, 0]
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    new, _ = run_step(genome, 'A1T,G2C')
    assert [c[8] for c, _ in new] == [1, 1]
    assert new[0][0][4] == new[0][0][6] == config.AA_VOCABS['S']


# ---------- ⑤ 文脈はステップ開始時のゲノムから ----------
def test_context_bases_come_from_the_genome_at_step_start(cfg):
    genome = 'ATGGCTTTACCC'
    cfg(FEATURE_ORDER_INDEPENDENT=False)
    legacy, _ = run_step(genome, 'A1T,G4C')
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    new, _ = run_step(genome, 'A1T,G4C')
    n = config.BASE_VOCABS['n']
    # G4(位置4)の左文脈: 位置-1..3 → [n,n,n,A(1),T(2),G(3)] の末尾5つ = [n,n,A,T,G]
    left_new = new[1][0][9:14]
    left_old = legacy[1][0][9:14]
    A, T, G = (config.BASE_VOCABS[b] for b in 'ATG')
    assert left_new == [n, n, A, T, G]                  # 位置1はステップ開始時の A のまま
    assert left_old == [n, n, T, T, G]                  # 従来は先に適用された A1T の T が入る


# ---------- ⑥⑦ 状態: 全変異適用後・復元 ----------
def test_state_after_step_has_all_applicable_mutations_applied(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = 'ATGGCTTTACCC'
    _, state = run_step(genome, 'A1T,G3C,T6A')
    assert "".join(state['base']) == 'TTCGCATTACCC'
    assert state['codon'][:3] == ['TTC'] * 3 and state['codon'][3:6] == ['GCA'] * 3


def test_reference_mismatch_mutation_is_not_applied_and_does_not_change_the_group(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = 'ATGGCTTTACCC'
    feats, state = run_step(genome, 'C1T,G3C')           # C1T は参照塩基(A)と不一致
    assert "".join(state['base']) == 'ATCGCTTTACCC'      # G3C のみ適用
    assert feats[0][0][4] == feats[0][0][6]              # 不一致の変異は「変化なし」(aa_before==aa_after)
    assert feats[1][0][6] == config.AA_VOCABS['I']       # ATG→ATC(Ile)。C1T が合成に混ざらない


def test_path_state_is_restored_after_processing(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = random_genome(random.Random(5), 12)
    state = make_state(genome)
    F.Feature_path_fast(['A1T,G3C', 'T2C,C4A', 'G7A'], state, freq_for(len(genome)), DISSIM, PAM, HOST)
    assert state == make_state(genome)


def test_later_steps_start_from_the_state_after_all_mutations_of_the_previous_step(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = 'ATGGCTTTACCC'
    path = F.Feature_path_fast(['A1T,G3C', 'T2C'], make_state(genome), freq_for(12), DISSIM, PAM, HOST)
    # 2ステップ目 T2C は TTC(Phe) を起点に TCC(Ser)
    c = path[1][0][0]
    assert c[4] == config.AA_VOCABS['F'] and c[6] == config.AA_VOCABS['S']


# ---------- 累積カウント ----------
def test_cumulative_counts_follow_the_shared_synonymous_flag(cfg):
    cfg(FEATURE_ORDER_INDEPENDENT=True)
    genome = 'AGTGCTTTACCC'
    path = F.Feature_path_fast(['A1T,G2C', 'T6A'], make_state(genome), freq_for(12), DISSIM, PAM, HOST)
    # 1ステップ目の2変異はどちらも同義 → 変異ごとに数えて cum_syn=2。2ステップ目に引き継がれる
    assert path[1][0][1][-2:] == [2.0, 0.0]


# ---------- ⑧ 前処理のラベル ----------
def test_preprocess_labels_use_the_shared_synonymous_flag(tmp_path, cfg, monkeypatch):
    from transformer import preprocess
    cfg(FEATURE_ORDER_INDEPENDENT=True, MAX_CO_OCCURRENCE=5, INCREMENTAL_CACHE_DIR=str(tmp_path / 'cache'),
        RAW_PATH_TRUNCATE_LEN=None)
    monkeypatch.setattr(preprocess, 'get_all_data_base_dirs', lambda: [str(tmp_path / 'usher')])
    d = tmp_path / 'usher' / 'X'
    d.mkdir(parents=True)
    (d / 'mutation_paths.tsv').write_text('name\tc\tpath\nS1\tx\tG7A>A1T,G2C\n')
    genome = 'AGTGCTTTACCC'
    paths, stats = preprocess.process_strain_features_core_chunked(
        'X', make_state(genome), freq_for(12), DISSIM, PAM, HOST, 'oi')
    (sample,) = preprocess.load_strain_cache(paths[0])
    y = sample[5]
    assert [t[4] for t in y] == [1, 1]                   # ターゲット2変異の同義ラベルが揃う（従来は 0, 0）
    assert [t[1] for t in y] == [1, 2]
