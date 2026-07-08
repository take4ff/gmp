# petra/tokenizer.py — 変異パスのトークン化
#
# v1: 変異1件=1フラットトークン（例 "A23403G"）。
# v2: 原論文(petra/PETra data_preprocess/utils.py の mutation_encoding, version=0)に
#     倣い、変異1件を複数トークンに展開する:
#       [mut, pos, nuc_mut, region, region_pos, region_mut]
#     原論文の9フィールド([mut,pos,nuc_mut,anno,anno_pos,anno_mut,
#     overlap_anno,overlap_anno_pos,overlap_anno_mut])のうち、overlap系3フィールドは
#     本体プロジェクトに対応する重複ORF注釈データが無いため省略した6フィールド版。
#     region(anno)は遺伝子領域そのもので、これにより region 単位の予測・評価が可能になる。
#
# 重要: v2は語彙が大きく変わる（v1の92,332から大幅増）ため、v1と同じキャッシュファイルを
# 絶対に上書きしないこと。既存の walk_forward 実行中プロセスが v1 キャッシュを再読込した際に
# 語彙サイズが変わってしまうと、fold間のチェックポイント引き継ぎ(state_dict)が壊れる。

import os
import re
import csv as _csv
import pickle

SPECIAL_TOKENS = ['[PAD]', '[BOS]', '[EOS]', '[UNK]', '[SEP]']
NUC_LETTERS = ['A', 'T', 'G', 'C']


def _sanitize(s):
    """クレード名などスペース・括弧を含む文字列をトークン化に安全な形式へ変換。"""
    return re.sub(r'[^A-Za-z0-9._-]', '_', s)


def _parse_mutations(raw_path):
    """raw_path を変異リストに展開（順序保持）。

    フォーマット: "mut1>mut2,mut3>mut4" — '>' が時系列区切り、',' が共起区切り
    """
    muts = []
    for node in raw_path.split('>'):
        for m in node.split(','):
            m = m.strip()
            if m:
                muts.append(m)
    return muts


_MUT_PARTS_RE = re.compile(r'^([ACGT])(\d+)([ACGT])$')


def _parse_mutation_parts(m: str):
    """"A23403G" -> ('A', 23403, 'G')。パースできなければ None。"""
    match = _MUT_PARTS_RE.match(m)
    if not match:
        return None
    ref, pos, alt = match.group(1), int(match.group(2)), match.group(3)
    return ref, pos, alt


def load_codon_annotation(csv_path: str) -> dict:
    """codon_mutation4.csv (base_pos,base,protein,protein_pos,codon,codon_pos,>A,>T,>G,>C)
    を base_pos -> {'protein', 'protein_pos', 'codon', 'to': {base: 変異後codon}} に変換する。
    """
    ann = {}
    with open(csv_path, newline='', encoding='utf-8-sig') as f:
        for row in _csv.DictReader(f):
            try:
                pos = int(row['base_pos'])
            except (KeyError, TypeError, ValueError):
                continue
            ann[pos] = {
                'protein': row.get('protein', 'noncoding'),
                'protein_pos': row.get('protein_pos', '0'),
                'codon': row.get('codon', 'none'),
                'to': {b: (row.get('>' + b) or 'none') for b in NUC_LETTERS},
            }
    return ann


class MutationTokenizer:
    """変異パス → トークンID列のエンコーダ。

    語彙は DB (+ v2 の場合は codon_mutation4.csv) から構築し、cache に保存・再利用する。
    """

    def __init__(self, use_region_fields: bool = False):
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []
        self.use_region_fields = use_region_fields
        self._ann: dict = {}          # base_pos -> 遺伝子注釈（use_region_fields=True のときのみ）
        self._dna2protein: dict = {}  # コドン -> アミノ酸

    # ------------------------------------------------------------------ #
    # 語彙構築
    # ------------------------------------------------------------------ #

    def build_from_db(self, db_path: str, chunk_size: int = 10000, codon_csv: str = None):
        """DB の全 raw_path をスキャンして語彙を構築する。

        use_region_fields=True の場合、codon_csv (codon_mutation4.csv) から
        region/region_pos/nuc_mut/region_mut トークンも合わせて構築する（DBスキャン不要、
        位置の全域をCSVから直接列挙する）。
        """
        import duckdb

        # 特殊トークン
        for tok in SPECIAL_TOKENS:
            self._add(tok)

        # 年・月トークン（SARS-CoV-2 の観測期間をカバー）
        for y in range(2019, 2027):
            self._add(f'[YEAR_{y}]')
        for m in range(1, 13):
            self._add(f'[MONTH_{m:02d}]')

        # クレードトークン
        con = duckdb.connect(db_path, read_only=True)
        clades = [r[0] for r in con.execute(
            'SELECT DISTINCT clade FROM strains WHERE clade IS NOT NULL').fetchall()]
        for c in sorted(clades):
            self._add(f'[CLADE_{_sanitize(c)}]')

        # 国トークン（country カラムが存在する場合のみ）
        cols = [r[1] for r in con.execute('PRAGMA table_info(samples)').fetchall()]
        if 'country' in cols:
            countries = [r[0] for r in con.execute(
                "SELECT DISTINCT country FROM samples WHERE country IS NOT NULL AND country != ''").fetchall()]
            for ct in sorted(countries):
                self._add(f'[COUNTRY_{_sanitize(ct)}]')

        # 変異トークン（全 raw_path をスキャン、flat mut トークンは v1/v2 共通）
        mutations: set[str] = set()
        total = con.execute('SELECT COUNT(*) FROM samples').fetchone()[0]
        offset = 0
        while offset < total:
            rows = con.execute(
                f'SELECT raw_path FROM samples LIMIT {chunk_size} OFFSET {offset}'
            ).fetchall()
            for (raw_path,) in rows:
                mutations.update(_parse_mutations(raw_path))
            offset += chunk_size
            if offset % 100000 == 0:
                print(f'  scanned {offset}/{total} samples, '
                      f'{len(mutations)} unique mutations so far')
        con.close()

        for m in sorted(mutations):
            self._add(m)

        if self.use_region_fields:
            if not codon_csv or not os.path.exists(codon_csv):
                raise FileNotFoundError(
                    f'use_region_fields=True には codon_csv が必要です: {codon_csv}')
            self._build_region_field_vocab(codon_csv)

        print(f'Vocabulary size: {self.vocab_size}')
        return self

    def _build_region_field_vocab(self, codon_csv: str):
        """pos/region/region_pos/nuc_mut/region_mut トークンを codon_mutation4.csv から
        直接列挙して追加する（DBスキャン不要、位置の全域をカバーできる）。
        """
        from transformer_260707.db.feature import DNA2Protein
        self._dna2protein = DNA2Protein
        self._ann = load_codon_annotation(codon_csv)
        print(f'  codon annotation loaded: {len(self._ann):,} positions')

        regions = set()
        for pos, info in self._ann.items():
            self._add(f'[POS_{pos}]')
            region = info['protein']
            regions.add(region)
            self._add(f"[REGIONPOS_{region}_{info['protein_pos']}]")

            codon_before = info['codon']
            aa_before = (DNA2Protein.get(codon_before.upper(), 'n')
                        if codon_before and codon_before != 'none' else 'n')
            for alt in NUC_LETTERS:
                codon_after = info['to'].get(alt, 'none')
                aa_after = (DNA2Protein.get(codon_after.upper(), 'n')
                           if codon_after and codon_after != 'none' else 'n')
                self._add(f'[REGIONMUT_{aa_before}{aa_after}]')
        for r in sorted(regions):
            self._add(f'[REGION_{r}]')
        self._add('[REGION_noncoding]')

        for ref in NUC_LETTERS:
            for alt in NUC_LETTERS:
                if ref != alt:
                    self._add(f'[NUCMUT_{ref}{alt}]')

    def _add(self, tok: str):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)

    # ------------------------------------------------------------------ #
    # エンコード / デコード
    # ------------------------------------------------------------------ #

    def _encode_mutation(self, m: str) -> list[int]:
        """1変異 -> トークンID列。use_region_fields=False なら [mut] の1トークン、
        True なら [mut, pos, nuc_mut, region, region_pos, region_mut] の6トークン。
        """
        unk = self.token2id['[UNK]']
        mut_id = self.token2id.get(m, unk)
        if not self.use_region_fields:
            return [mut_id]

        parts = _parse_mutation_parts(m)
        if parts is None:
            return [mut_id, unk, unk, unk, unk, unk]
        ref, pos, alt = parts

        info = self._ann.get(pos)
        pos_id = self.token2id.get(f'[POS_{pos}]', unk)
        nucmut_id = self.token2id.get(f'[NUCMUT_{ref}{alt}]', unk)

        if info is None:
            return [mut_id, pos_id, nucmut_id, unk, unk, unk]

        region = info['protein']
        region_id = self.token2id.get(f'[REGION_{region}]', unk)
        regionpos_id = self.token2id.get(f"[REGIONPOS_{region}_{info['protein_pos']}]", unk)

        codon_before = info['codon']
        aa_before = (self._dna2protein.get(codon_before.upper(), 'n')
                    if codon_before and codon_before != 'none' else 'n')
        codon_after = info['to'].get(alt, 'none')
        aa_after = (self._dna2protein.get(codon_after.upper(), 'n')
                   if codon_after and codon_after != 'none' else 'n')
        regionmut_id = self.token2id.get(f'[REGIONMUT_{aa_before}{aa_after}]', unk)

        return [mut_id, pos_id, nucmut_id, region_id, regionpos_id, regionmut_id]

    def encode(self, raw_path: str, collection_date=None, clade=None,
               country=None, max_history_steps: int = None) -> list[int]:
        """1サンプルをトークンID列に変換する。

        形式: [BOS] [YEAR_?] [MONTH_??] [COUNTRY_?] [CLADE_?] mut1(...) mut2(...) ... [EOS]
        use_region_fields=True の場合、各 mut は [mut,pos,nuc_mut,region,region_pos,region_mut]
        の6トークンに展開される。

        max_history_steps: 本体(transformer_260707.db.dataset)の
        `raw_path.split('>')[-MAX_SEQ_LEN:]` と同じ切り出し方で、直近何タイムステップ
        （'>'区切り、各タイムステップ内の共起変異はそのまま維持）のみを使うか。
        None なら系統樹の根からの全履歴を使う（従来動作）。
        """
        if max_history_steps is not None:
            nodes = raw_path.split('>')
            if len(nodes) > max_history_steps:
                raw_path = '>'.join(nodes[-max_history_steps:])

        unk = self.token2id['[UNK]']
        tokens = [self.token2id['[BOS]']]

        # 時刻コンテキスト
        if collection_date:
            parts = str(collection_date).split('-')
            if len(parts) >= 1:
                tokens.append(self.token2id.get(f'[YEAR_{parts[0]}]', unk))
            if len(parts) >= 2:
                tokens.append(self.token2id.get(f'[MONTH_{parts[1]}]', unk))

        # 国コンテキスト
        if country:
            tokens.append(self.token2id.get(f'[COUNTRY_{_sanitize(country)}]', unk))

        # クレードコンテキスト
        if clade:
            tokens.append(self.token2id.get(f'[CLADE_{_sanitize(clade)}]', unk))

        # 変異列
        for m in _parse_mutations(raw_path):
            tokens.extend(self._encode_mutation(m))

        tokens.append(self.token2id['[EOS]'])
        return tokens

    def decode(self, ids: list[int]) -> list[str]:
        return [self.id2token[i] if i < len(self.id2token) else '[UNK]' for i in ids]

    # ------------------------------------------------------------------ #
    # 特殊トークン ID
    # ------------------------------------------------------------------ #

    @property
    def pad_id(self) -> int:
        return self.token2id['[PAD]']

    @property
    def bos_id(self) -> int:
        return self.token2id['[BOS]']

    @property
    def eos_id(self) -> int:
        return self.token2id['[EOS]']

    @property
    def vocab_size(self) -> int:
        return len(self.id2token)

    def is_context_token(self, tok_id: int) -> bool:
        """BOS/EOS/PAD/時刻/国/クレードトークンなら True（評価時にスキップ対象）。"""
        tok = self.id2token[tok_id] if tok_id < len(self.id2token) else ''
        return (tok in ('[PAD]', '[BOS]', '[EOS]', '[UNK]', '[SEP]') or
                tok.startswith('[YEAR_') or
                tok.startswith('[MONTH_') or
                tok.startswith('[COUNTRY_') or
                tok.startswith('[CLADE_'))

    def is_region_token(self, tok_id: int) -> bool:
        """[REGION_xxx] トークンなら True（region単位の精度評価に使用）。"""
        tok = self.id2token[tok_id] if tok_id < len(self.id2token) else ''
        return tok.startswith('[REGION_')

    def is_position_token(self, tok_id: int) -> bool:
        """[POS_xxx] トークンなら True（位置単位の精度評価に使用）。"""
        tok = self.id2token[tok_id] if tok_id < len(self.id2token) else ''
        return tok.startswith('[POS_')

    # ------------------------------------------------------------------ #
    # 保存 / 読み込み
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'token2id': self.token2id, 'id2token': self.id2token,
                        'use_region_fields': self.use_region_fields}, f)
        print(f'Tokenizer saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'MutationTokenizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tok = cls(use_region_fields=data.get('use_region_fields', False))
        tok.token2id = data['token2id']
        tok.id2token = data['id2token']
        if tok.use_region_fields:
            from transformer_260707 import config as main_config
            tok._dna2protein = __import__(
                'transformer_260707.db.feature', fromlist=['DNA2Protein']).DNA2Protein
            tok._ann = load_codon_annotation(main_config.CODON_CSV)
        return tok
