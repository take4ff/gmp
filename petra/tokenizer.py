# petra/tokenizer.py — 変異パスのトークン化

import os
import re
import pickle

SPECIAL_TOKENS = ['[PAD]', '[BOS]', '[EOS]', '[UNK]', '[SEP]']


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


class MutationTokenizer:
    """変異パス → トークンID列のエンコーダ。

    語彙は DB から動的構築し、cache に保存・再利用する。
    """

    def __init__(self):
        self.token2id: dict[str, int] = {}
        self.id2token: list[str] = []

    # ------------------------------------------------------------------ #
    # 語彙構築
    # ------------------------------------------------------------------ #

    def build_from_db(self, db_path: str, chunk_size: int = 10000):
        """DB の全 raw_path をスキャンして語彙を構築する。"""
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

        # 変異トークン（全 raw_path をスキャン）
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

        print(f'Vocabulary size: {self.vocab_size}')
        return self

    def _add(self, tok: str):
        if tok not in self.token2id:
            self.token2id[tok] = len(self.id2token)
            self.id2token.append(tok)

    # ------------------------------------------------------------------ #
    # エンコード / デコード
    # ------------------------------------------------------------------ #

    def encode(self, raw_path: str, collection_date=None, clade=None,
               country=None) -> list[int]:
        """1サンプルをトークンID列に変換する。

        形式: [BOS] [YEAR_?] [MONTH_??] [COUNTRY_?] [CLADE_?] mut1 mut2 ... [EOS]
        """
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
            tokens.append(self.token2id.get(m, unk))

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

    # ------------------------------------------------------------------ #
    # 保存 / 読み込み
    # ------------------------------------------------------------------ #

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'token2id': self.token2id, 'id2token': self.id2token}, f)
        print(f'Tokenizer saved to {path}')

    @classmethod
    def load(cls, path: str) -> 'MutationTokenizer':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tok = cls()
        tok.token2id = data['token2id']
        tok.id2token = data['id2token']
        return tok
