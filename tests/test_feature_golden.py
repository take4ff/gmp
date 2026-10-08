"""B2: db/feature.py の特徴量計算（モデル入力そのもの）を、手計算できる極小ゲノムで検証する。

極小ゲノム(12塩基): ATG GCT TTA CCC  → Met Ala Leu Pro（全て nsp1、aa_pos 1..4、codon_pos 1..3）。
過去の静かなバグ（raw_path切り詰め等）と同種の「もっともらしい値を出し続ける誤り」を検知する。
"""
import copy

import pytest

from transformer import config
from transformer.db import feature as F

BASES = list('ATGGCTTTACCC')
CODONS = ['ATG'] * 3 + ['GCT'] * 3 + ['TTA'] * 3 + ['CCC'] * 3


def make_state():
    return {'base': list(BASES), 'protein': ['nsp1'] * 12,
            'aa_pos': [1] * 3 + [2] * 3 + [3] * 3 + [4] * 3,
            'codon': list(CODONS), 'codon_pos': [1, 2, 3] * 4}


FREQ = {'A->T': [0.5] * 12, 'T->A': [0.25] * 12, 'T->C': [0.125] * 12, 'G->C': [0.0625] * 12}
DISSIM = {('M', 'L'): {'hydro': 1.0, 'charge': 0.0, 'size': -1.5, 'blsm': 2.0},
          ('L', 'S'): {'hydro': -2.0, 'charge': 0.0, 'size': 0.5, 'blsm': -1.0}}
PAM = {('M', 'L'): 3.0, ('L', 'S'): -2.0}
AA = config.AA_VOCABS
BASE = config.BASE_VOCABS
NSP1 = config.PROTEIN_VOCABS['nsp1']


def feats(path, state=None):
    state = make_state() if state is None else state
    return F.Feature_path_fast(path, state, FREQ, DISSIM, PAM, {})


def test_nonsynonymous_mutation_features_by_hand():
    (cat, num), = feats(['A1T'])[0]
    # A1T: ATG(Met)→TTG(Leu)。位置1(codon_pos=1)、aa_pos=1
    assert cat[:9] == [BASE['A'], 1, BASE['T'], 1, AA['M'], 1, AA['L'], NSP1, 0]
    left, right = cat[9:14], cat[14:19]                       # CONTEXT_WINDOW=5
    assert left == [BASE['n']] * 5                            # ゲノム端は 'n'
    assert right == [BASE['T'], BASE['G'], BASE['G'], BASE['C'], BASE['T']]    # 位置2..6 = T G G C T
    assert num[0] == pytest.approx(0.5)                       # A->T の再発頻度
    assert num[1:6] == pytest.approx([1.0, 0.0, -1.5, 2.0, 3.0])   # hydro, charge, size, blsm, pam250
    assert num[-2:] == [0.0, 0.0]                             # 累積 syn / nonsyn（この時点では0）
    assert len(num) == 6 + len(F._ADAPT_ZERO) + 2


def test_synonymous_mutation_is_flagged():
    (cat, num), = feats(['T6A'])[0]                           # GCT→GCA（Ala→Ala）
    assert cat[3] == 3 and cat[4] == cat[6] == AA['A'] and cat[8] == 1


def test_context_near_genome_end_is_padded_with_n():
    (cat, _), = feats(['C12A'])[0]
    assert cat[14:19] == [BASE['n']] * 5                      # 右側はゲノム外
    # 左5塩基 = 位置7..11 = T T A C C（遠い方→近い方の順）
    assert cat[9:14] == [BASE['T'], BASE['T'], BASE['A'], BASE['C'], BASE['C']]


def test_reference_mismatch_does_not_change_state():
    state = make_state()
    feats(['C1T'], state)                                     # 参照塩基は A なので適用されない
    assert state == make_state()


def test_state_is_fully_restored_after_path():
    state = make_state()
    feats(['A1T', 'T2C', 'G3C', 'T6A'], state)
    assert state == make_state()                              # 共有状態は finally で完全復元


def test_sequential_mutations_see_updated_codon():
    ts1, ts2 = feats(['A1T', 'T2C'])
    (cat1, num1), = ts1
    (cat2, num2), = ts2
    assert cat1[4:7] == [AA['M'], 1, AA['L']]
    # 2ステップ目は更新後のコドン TTG(Leu) を起点に TCG(Ser)
    assert cat2[3] == 2 and cat2[4] == AA['L'] and cat2[6] == AA['S'] and cat2[8] == 0
    assert num2[-2:] == [0.0, 1.0]                            # 直前までの累積: nonsyn=1
    assert num2[1:6] == pytest.approx([-2.0, 0.0, 0.5, -1.0, -2.0])


def test_cumulative_counts_track_synonymous_and_nonsynonymous():
    _, ts2, ts3 = feats(['T6A', 'A1T', 'T2C'])
    assert ts2[0][1][-2:] == [1.0, 0.0]                       # syn=1, nonsyn=0
    assert ts3[0][1][-2:] == [1.0, 1.0]


def test_cooccurring_mutations_share_one_timestep():
    path = feats(['A1T,T6A'])
    assert len(path) == 1 and len(path[0]) == 2
    assert [c[1] for c, _ in path[0]] == [1, 6]


def test_cooccurring_mutations_in_the_same_codon_depend_on_listed_order():
    """【現状仕様の記録】同一コドン内の共起変異は、記載順に適用され、後の変異のaa_before/afterが前の変異の
    影響を受ける。モデルは共起集合を順序なしで集約するが、特徴量生成の段階では順序依存が残る。
    変更する場合は意図的な仕様変更として、このテストを更新すること。"""
    (a1, _), (a2, _) = feats(['A1T,G3C'])[0]
    (b1, _), (b2, _) = feats(['G3C,A1T'])[0]
    assert a2[4] == AA['L']                       # A1T の後なので G3C の元コドンは TTG(Leu)
    assert b1[4] == AA['M']                       # G3C が先なら元コドンは ATG(Met)
    assert a2[4] != b1[4]


def test_filter_co_occur():
    assert F.filter_co_occur(['A1T', 'A1T,T2C'], 2) is True
    assert F.filter_co_occur(['A1T', 'A1T,T2C,G3C'], 2) is False


def test_features_are_deterministic_and_do_not_mutate_inputs():
    freq, state = copy.deepcopy(FREQ), make_state()
    a = F.Feature_path_fast(['A1T', 'T2C'], state, freq, DISSIM, PAM, {})
    b = F.Feature_path_fast(['A1T', 'T2C'], state, freq, DISSIM, PAM, {})
    assert a == b and freq == FREQ
