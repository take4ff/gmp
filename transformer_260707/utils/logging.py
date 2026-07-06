# --- utils/logging.py ---
# ログ出力・評価指標計算

import time
from .. import config


def force_print(message):
    """タイムスタンプ付きでメッセージを即時フラッシュ出力する。"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{timestamp}] {message}", flush=True)


def print_config():
    """現在の設定値を一覧表示する。"""
    force_print("Current Configurations:")
    for attr in dir(config):
        if not attr.startswith("__"):
            value = getattr(config, attr)
            print(f"  {attr}: {value}")


def print_sample_structure(sample, sample_idx=0):
    """1サンプルのデータ構造をCSV形式で表示してモデルの入力形式を確認する。

    Args:
        sample: (x, y_targets, original_len, raw_path, strain, strength_score) のタプル
        sample_idx: 表示用のサンプル番号
    """
    x, y_targets, original_len, raw_path, strain, strength_score = sample

    print("\n" + "=" * 60)
    print(f"【サンプル {sample_idx} のデータ構造 (CSV形式)】")
    print("=" * 60)

    print(f"\n# メタ情報")
    print(f"strain,{strain}")
    print(f"strength_score,{strength_score:.3f}")
    print(f"original_len,{original_len}")
    print(f"x_len,{len(x)}")
    print(f"raw_path,{raw_path}")

    print(f"\n# ターゲット (Y)")
    print("target_idx,region_id,position_id,aa_pos_id,codon_pos_id,is_synonymous")
    for i, target in enumerate(y_targets):
        region_id, position_id, aa_pos_id, codon_pos_id, is_synonymous = target
        print(f"{i},{region_id},{position_id},{aa_pos_id},{codon_pos_id},{is_synonymous}")

    print(f"\n# 入力特徴量 (X)")
    print("# cat: bef=塩基前, pos=塩基位置, aft=塩基後, c_pos=コドン位置, aa_b=アミノ酸前, p_pos=アミノ酸配列位置, aa_a=アミノ酸後, region=領域")
    print("# num: hydro=疎水性変化, size=サイズ変化, charge=電荷変化, dissim=類似度, pam=PAM250, freq=頻度")
    print("timestep,co_occur,bef,pos,aft,c_pos,aa_b,p_pos,aa_a,region,freq,hydro,charge,size,dissim,pam")

    for t_idx, timestep in enumerate(x):
        for c_idx, event in enumerate(timestep):
            cat_features, num_features = event

            if all(f == 0 for f in cat_features):
                continue

            cat_str = ",".join(str(f) for f in cat_features)
            num_str = ",".join(f"{f:.3f}" for f in num_features)
            print(f"{t_idx},{c_idx},{cat_str},{num_str}")

    print("=" * 60 + "\n")


def calculate_metrics(pred_sets, target_sets):
    """Hit Rate・Precision・Recall・F1スコアを計算する。

    Args:
        pred_sets: list of set (予測集合のリスト)
        target_sets: list of set (正解集合のリスト)

    Returns:
        Tuple[float, float, float, float]: (hit_rate, precision, recall, f1) in %
    """
    total_samples = len(pred_sets)
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    hits = 0
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    valid_samples = 0

    for pred_s, target_s in zip(pred_sets, target_sets):
        if not target_s:
            continue

        valid_samples += 1

        # Intersection (TP)
        tp = len(pred_s.intersection(target_s))

        if tp > 0:
            hits += 1

        precision = tp / len(pred_s) if len(pred_s) > 0 else 0
        recall = tp / len(target_s) if len(target_s) > 0 else 0

        if (precision + recall) > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0

        total_precision += precision
        total_recall += recall
        total_f1 += f1

    if valid_samples == 0:
        return 0.0, 0.0, 0.0, 0.0

    avg_hit_rate = (hits / valid_samples) * 100
    avg_precision = (total_precision / valid_samples) * 100
    avg_recall = (total_recall / valid_samples) * 100
    avg_f1 = (total_f1 / valid_samples) * 100

    return avg_hit_rate, avg_precision, avg_recall, avg_f1


def calculate_weighted_macro_recall(pred_sets, target_sets):
    """Weighted Recall@1 と Macro Recall@1 をクラス単位で計算する。

    各サンプルについて「正解クラス c を予測できたか (0/1)」を集計し、
    - Weighted Recall@1: Σ(tp_c) / Σ(count_c)  (正解頻度に比例した重み付け)
    - Macro Recall@1  : mean over classes of (tp_c / count_c)

    TOP_K_EVAL=1 を前提とするが、複数正解 (共起) にも対応する。
    共起の場合、いずれか1クラスに正解すればそのクラスの TP を加算する。

    Args:
        pred_sets: list of set — 各サンプルの予測クラス集合
        target_sets: list of set — 各サンプルの正解クラス集合

    Returns:
        Tuple[float, float]: (weighted_recall_pct, macro_recall_pct)
    """
    # クラスごとに (tp_count, total_count) を集計
    class_tp = {}    # class_id -> True Positive 数
    class_cnt = {}   # class_id -> 正解ラベルとして出現した総数

    for pred_s, target_s in zip(pred_sets, target_sets):
        if not target_s:
            continue
        hit = len(pred_s.intersection(target_s)) > 0
        for c in target_s:
            class_cnt[c] = class_cnt.get(c, 0) + 1
            if hit and c in pred_s:
                class_tp[c] = class_tp.get(c, 0) + 1

    if not class_cnt:
        return 0.0, 0.0

    total_count = sum(class_cnt.values())

    # Weighted Recall@1
    weighted_tp = sum(class_tp.get(c, 0) for c in class_cnt)
    weighted_recall = (weighted_tp / total_count * 100) if total_count > 0 else 0.0

    # Macro Recall@1
    per_class_recalls = [class_tp.get(c, 0) / cnt for c, cnt in class_cnt.items()]
    macro_recall = (sum(per_class_recalls) / len(per_class_recalls) * 100) if per_class_recalls else 0.0

    return weighted_recall, macro_recall


def init_wandb():
    """WandBを初期化して wandb モジュールを返す（失敗時は None）。"""
    import os
    import types
    from datetime import datetime
    try:
        import wandb
        if config.USE_WANDB:
            if getattr(config, 'WANDB_OFFLINE', False):
                os.environ["WANDB_MODE"] = "offline"
                force_print("[INFO] WandB: オフラインモードで起動")

            config_dict = {
                k: v for k, v in vars(config).items()
                if not k.startswith('__') and not isinstance(v, types.ModuleType)
            }
            wandb.init(
                project=config.WANDB_PROJECT_NAME,
                name=config.WANDB_RUN_NAME if config.WANDB_RUN_NAME else f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=config_dict
            )
            return wandb
    except ImportError:
        force_print("[WARNING] wandb not available")
    return None


def finish_wandb(wandb_module):
    """WandBセッションを終了し、オフラインモードの場合は自動でsyncを実行する。

    Args:
        wandb_module: init_wandb()で返されたwandbモジュール
    """
    if wandb_module is None or not config.USE_WANDB:
        return

    try:
        import subprocess
        import os

        run_dir = None
        if wandb_module.run is not None:
            run_dir = wandb_module.run.dir

        wandb_module.finish()
        force_print("[INFO] WandB: セッション終了")

        if getattr(config, 'WANDB_OFFLINE', False) and run_dir:
            run_path = os.path.dirname(run_dir)
            force_print(f"[INFO] WandB: オフラインデータを同期中... ({run_path})")

            try:
                result = subprocess.run(
                    ['wandb', 'sync', run_path],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    force_print("[INFO] WandB: 同期完了")
                else:
                    force_print(f"[WARNING] WandB sync failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                force_print("[WARNING] WandB sync timed out (5 min). Run manually: wandb sync ./wandb/run-*")
            except Exception as e:
                force_print(f"[WARNING] WandB sync error: {e}")

    except Exception as e:
        force_print(f"[WARNING] WandB finish error: {e}")
