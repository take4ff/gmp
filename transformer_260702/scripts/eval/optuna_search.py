"""Optuna ハイパラ探索エントリポイント。

- ローカル SQLite に保存 → オフライン・再開可能（長時間学習でオンライン接続が切れても影響なし）。
- 1 trial = 1 学習（main() を呼ぶ）。目的関数は run_summary.json の val 指標（既定: 塩基位置 Top-1 hit-rate）。
- pruning: 各エポックの early-stop 指標を報告し、見込みの薄い trial を早期打ち切り（計算節約）。
- 探索対象（高レバレッジに限定）:
    LEARNING_RATE / DROPOUT / LOSS_WEIGHT_{REGION,POSITION,AA_POS}
    ※ loss weight は「position↑/領域↓」トレードオフを直接制御する。

Usage:
    pip install optuna    # 未導入の場合
    python -m transformer_260702.scripts.eval.optuna_search --n-trials 30
    # 中断後の再開: 同じ --study-name / --storage を指定すれば続きから
    # 特定パラメータのみ探索: --params lr dropout

出力:
    outputs/transformer_260702/scripts/optuna/<study_name>.db          （SQLite: 全 trial 履歴）
    outputs/transformer_260702/scripts/optuna/<study_name>_best.json    （ベスト構成）
各 trial の学習結果: outputs/.../results/optuna/<study_name>/trial_<n>/...

pruning を効かせるには config で EARLY_STOPPING_METRIC を hit-rate・MODE='max'・
OPTUNA_DIRECTION='maximize' に揃えること（方向一致時のみ中間報告する）。
"""

import argparse
import glob
import json
import os


# 探索対象の定義: name -> (config属性, suggest 関数)
def _suggest_space(trial, params):
    from transformer_260702 import config
    if 'lr' in params:
        config.LEARNING_RATE = trial.suggest_float('learning_rate', 1e-5, 3e-3, log=True)
    if 'dropout' in params:
        config.DROPOUT = trial.suggest_float('dropout', 0.0, 0.3)
    if 'loss_weights' in params:
        config.LOSS_WEIGHT_REGION = trial.suggest_float('lw_region', 0.05, 1.0)
        config.LOSS_WEIGHT_POSITION = trial.suggest_float('lw_position', 0.1, 1.0)
        config.LOSS_WEIGHT_AA_POS = trial.suggest_float('lw_aa_pos', 0.02, 0.5)


def _read_objective(best_model_path, split, key):
    """trial の run_summary.json から目的指標を読む。"""
    if not best_model_path:
        return None
    run_dir = os.path.dirname(os.path.dirname(best_model_path))
    hits = glob.glob(os.path.join(run_dir, '**', 'run_summary.json'), recursive=True)
    if not hits:
        return None
    with open(hits[0]) as f:
        summary = json.load(f)
    return summary.get(split, {}).get(key)


def _make_objective(params):
    def objective(trial):
        import optuna
        from transformer_260702 import config
        from transformer_260702.main import main

        _suggest_space(trial, params)

        # 探索中はシード固定・出力を trial ごとに分離
        config.SEED = getattr(config, 'OPTUNA_SEED', 42)
        study_name = getattr(config, 'OPTUNA_STUDY_NAME', 'gmp_hpo')
        config.EXPERIMENT_NAME = f'optuna/{study_name}/trial_{trial.number}'

        # 学習（pruning 時は run_training が optuna.TrialPruned を送出 → ここまで伝播）
        best_model_path = main(optuna_trial=trial)

        split = getattr(config, 'OPTUNA_OBJECTIVE_SPLIT', 'val')
        key = getattr(config, 'OPTUNA_OBJECTIVE_KEY', 'top1_position_hit_rate_pct')
        value = _read_objective(best_model_path, split, key)
        if value is None:
            raise optuna.TrialPruned()  # 目的指標が読めない → 枝刈り扱い
        trial.set_user_attr('objective_metric', f'{split}.{key}')
        if best_model_path:
            trial.set_user_attr('run_dir', os.path.dirname(os.path.dirname(best_model_path)))
        return value

    return objective


def main_cli():
    try:
        import optuna
    except ImportError:
        raise SystemExit("optuna が未インストールです。`pip install optuna` を実行してください。")

    from transformer_260702 import config

    parser = argparse.ArgumentParser(description='Optuna hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=None,
                        help='試行数（省略時は config.OPTUNA_N_TRIALS）')
    parser.add_argument('--study-name', type=str, default=None)
    parser.add_argument('--storage', type=str, default=None,
                        help='SQLite URL 等（省略時は outputs/.../optuna/<study>.db）')
    parser.add_argument('--direction', type=str, default=None,
                        choices=['maximize', 'minimize'])
    parser.add_argument('--params', nargs='+', default=['lr', 'dropout', 'loss_weights'],
                        choices=['lr', 'dropout', 'loss_weights'],
                        help='探索対象（既定: 全部）')
    args = parser.parse_args()

    n_trials = args.n_trials or getattr(config, 'OPTUNA_N_TRIALS', 30)
    study_name = args.study_name or getattr(config, 'OPTUNA_STUDY_NAME', 'gmp_hpo')
    direction = args.direction or getattr(config, 'OPTUNA_DIRECTION', 'maximize')
    if args.study_name:
        config.OPTUNA_STUDY_NAME = study_name
    config.OPTUNA_DIRECTION = direction

    # SQLite 保存先（オフライン・再開可能）
    out_dir = os.path.join('outputs', 'transformer_260702', 'scripts', 'optuna')
    os.makedirs(out_dir, exist_ok=True)
    storage = args.storage or getattr(config, 'OPTUNA_STORAGE', None) \
        or f"sqlite:///{os.path.abspath(os.path.join(out_dir, study_name + '.db'))}"

    # 悪い trial を早期打ち切り（3 エポック warmup してから中央値枝刈り）
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name, storage=storage, direction=direction,
        pruner=pruner, load_if_exists=True,
    )

    print(f"[Optuna] study='{study_name}' storage='{storage}' direction={direction} "
          f"n_trials={n_trials} params={args.params}")
    print(f"[Optuna] objective = {getattr(config, 'OPTUNA_OBJECTIVE_SPLIT', 'val')}."
          f"{getattr(config, 'OPTUNA_OBJECTIVE_KEY', 'top1_position_hit_rate_pct')}")

    study.optimize(_make_objective(args.params), n_trials=n_trials)

    print("\n[Optuna] ===== 完了 =====")
    print(f"  best value : {study.best_value:.4f}")
    print(f"  best params: {study.best_params}")

    best_path = os.path.join(out_dir, f'{study_name}_best.json')
    with open(best_path, 'w') as f:
        json.dump({
            'best_value': study.best_value,
            'best_params': study.best_params,
            'best_trial_number': study.best_trial.number,
            'n_trials': len(study.trials),
            'direction': direction,
            'objective': f"{getattr(config, 'OPTUNA_OBJECTIVE_SPLIT', 'val')}."
                         f"{getattr(config, 'OPTUNA_OBJECTIVE_KEY', 'top1_position_hit_rate_pct')}",
        }, f, indent=2, ensure_ascii=False)
    print(f"  saved: {best_path}")


if __name__ == '__main__':
    main_cli()
