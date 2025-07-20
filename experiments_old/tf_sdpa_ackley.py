import argparse
import datetime
import json
import logging
import os
import signal
import sys
import time
from functools import partial

import random
import numpy as np
import optuna
from optuna.samplers import TPESampler, RandomSampler

from _src import set_logger
from _src import TFSdpaSampler


# シンプルなAckley関数の実装
class AckleyFunction:
    def __init__(self):
        self.min_value = -32
        self.max_value = 32
        
    def evaluate(self, x):
        # Ackley関数の実装
        a, b, c = 20, 0.2, 2 * np.pi
        term1 = -a * np.exp(-b * np.sqrt(np.mean(x ** 2)))
        term2 = -np.exp(np.mean(np.cos(c * x)))
        return term1 + term2 + a + np.exp(1)


# グローバル変数: 終了フラグとStudyオブジェクト
timeout_flag = False
current_study = None
save_info = {}


# タイムアウトハンドラ
def timeout_handler(signum, frame):
    global timeout_flag
    logging.warning("タイムアウトが発生しました。実験を終了します。")
    timeout_flag = True
    if current_study:
        save_results(forced=True)


# 結果を保存する関数
def save_results(forced=False):
    global current_study, save_info, timeout_flag
    
    if not current_study or not save_info:
        return
    
    # 時間切れによる強制終了時のメッセージ
    if forced:
        logging.warning("タイムアウトのため、現在の最良結果を保存します")
    
    # 結果出力
    logging.info(f"Best value: {current_study.best_value}")
    logging.info(f"Best params: {current_study.best_params}")
    
    # 最適解の座標を表示
    try:
        best_x = np.array([current_study.best_params[f"x_{i}"] for i in range(save_info["n_dim"])])
        logging.info(f"Best coordinates: {best_x}")
        
        # 結果をまとめたJSONを保存
        result_summary = {
            "study_name": save_info["name"],
            "seed": save_info["seed"],
            "cat_num": save_info["cat_num"],
            "max_radius": save_info["max_radius"], 
            "n_trials": save_info["n_trials"],
            "completed_trials": len(current_study.trials),
            "terminated_by_timeout": forced,
            "n_dim": save_info["n_dim"],
            "sampler": save_info["sampler_type"],
            "best_value": float(current_study.best_value),
            "best_params": {k: int(v) for k, v in current_study.best_params.items()},
            "best_coordinates": [int(x) for x in best_x],
            "total_points": save_info["total_points"],
            "valid_points": save_info["valid_points"],
            "valid_ratio": float(save_info["valid_ratio"]),
            "storage_path": save_info["storage"],
            "execution_time": time.time() - save_info["start_time"]
        }
        
        summary_path = os.path.join(save_info["results_dir"], f"{save_info['log_filename_base']}_summary.json")
        with open(summary_path, "w") as f:
            json.dump(result_summary, f, indent=2)
        
        logging.info(f"Saved result summary to {summary_path}")
        
        # 可視化結果を保存（オプション）
        if save_info.get("plot_save_dir"):
            try:
                fig = optuna.visualization.plot_optimization_history(current_study)
                plot_path = os.path.join(
                    save_info["plot_save_dir"], 
                    f"{save_info['name']}_optimization_history.png"
                )
                os.makedirs(os.path.dirname(plot_path), exist_ok=True)
                fig.write_image(plot_path)
                logging.info(f"Saved optimization history plot to {plot_path}")
                
                if save_info["n_dim"] <= 2:
                    fig = optuna.visualization.plot_contour(current_study)
                    plot_path = os.path.join(
                        save_info["plot_save_dir"], 
                        f"{save_info['name']}_contour.png"
                    )
                    fig.write_image(plot_path)
                    logging.info(f"Saved contour plot to {plot_path}")
            except Exception as e:
                logging.warning(f"Failed to save visualization: {e}")
    
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")


# 目的関数
def objective(trial, cat_num=None, n_dim=2, objective_function=None):
    """目的関数（Ackley関数）"""
    # タイムアウトチェック
    global timeout_flag
    if timeout_flag:
        # タイムアウト発生時は例外を発生させて最適化を停止
        raise optuna.exceptions.TrialPruned("Timeout occurred")
    
    # カテゴリリストの作成
    left = 0 - cat_num//2
    right = cat_num//2 if cat_num % 2 == 0 else (cat_num//2)
    categories = list(range(left, right + 1))
    
    # カテゴリ変数の提案
    x = np.array([trial.suggest_categorical(f"x_{i}", categories) for i in range(n_dim)])
    
    # 評価
    return objective_function.evaluate(x)


def create_constraint_tensor(cat_num, n_dim=2, max_radius=2):
    """カテゴリ数に基づいた制約テンソルを作成"""
    # カテゴリリストの作成
    left = 0 - cat_num//2
    right = cat_num//2 if cat_num % 2 == 0 else (cat_num//2)
    categories = list(range(left, right + 1))
    
    # カテゴリの数に基づいてテンソルサイズを決定
    tensor_shape = tuple([cat_num] * n_dim)
    constraint_tensor = np.zeros(tensor_shape, dtype=int)
    
    # 制約テンソルの値を設定（原点からの半径以内の点を有効とする）
    indices = np.indices(tensor_shape)
    for idx in np.ndindex(tensor_shape):
        # インデックスを座標値に変換
        coords = [categories[indices[dim][idx]] for dim in range(n_dim)]
        # 原点からの距離の二乗を計算
        r_squared = sum([x**2 for x in coords])
        if r_squared <= max_radius**2:
            constraint_tensor[idx] = 1
    
    return constraint_tensor, categories


# Optunaのコールバック関数: 定期的な進捗チェック
class TimeoutCallback:
    def __init__(self, timeout_seconds):
        self.start_time = time.time()
        self.timeout_seconds = timeout_seconds
    
    def __call__(self, study, trial):
        global timeout_flag
        elapsed_time = time.time() - self.start_time
        # 経過時間がタイムアウト時間を超えたら、タイムアウトフラグを立てる
        if elapsed_time > self.timeout_seconds:
            logging.warning(f"タイムアウト: {elapsed_time:.1f}秒経過")
            timeout_flag = True


def run_bo(settings):
    """ベイズ最適化を実行する関数"""
    global current_study, save_info, timeout_flag
    
    # 保存用情報を設定
    save_info = settings.copy()
    save_info["start_time"] = time.time()
    
    # シード設定
    random.seed(settings['seed'])
    np.random.seed(settings['seed'])
    
    # 制約テンソルを作成
    constraint_tensor, categories = create_constraint_tensor(
        settings["cat_num"], 
        settings["n_dim"], 
        settings["max_radius"]
    )
    
    # 探索空間の情報をログ出力
    logging.info(f"Search space: {settings['n_dim']}D with categories {categories}")
    logging.info(f"Category count: {len(categories)}")
    logging.info(f"Total search space size: {len(categories)**settings['n_dim']} points")
    
    # 制約情報をログ出力
    logging.info(f"Constraint tensor shape: {constraint_tensor.shape}")
    if settings["n_dim"] <= 2:
        logging.info(f"Constraint tensor:\n{constraint_tensor}")
    valid_count = np.sum(constraint_tensor)
    valid_ratio = valid_count/(len(categories)**settings['n_dim'])
    logging.info(f"Valid points: {valid_count}/{len(categories)**settings['n_dim']} ({valid_ratio*100:.1f}%)")
    
    # 保存情報に追加
    save_info["total_points"] = len(categories)**settings['n_dim']
    save_info["valid_points"] = int(valid_count)
    save_info["valid_ratio"] = float(valid_ratio)
    
    # Ackley関数インスタンスの準備
    objective_function = AckleyFunction()
    objective_with_args = partial(
        objective, 
        cat_num=settings["cat_num"], 
        n_dim=settings["n_dim"], 
        objective_function=objective_function
    )

    # サンプラー取得
    sampler = settings["sampler"]
    direction = "minimize"  # Ackley関数は最小化問題

    # Optuna Studyの作成
    study = optuna.create_study(
        study_name=settings["name"],
        sampler=sampler,
        direction=direction,
        storage=settings["storage"],
    )
    
    # グローバル変数に現在のStudyを設定
    current_study = study

    logging.info(f"Created new study '{settings['name']}' in {settings['storage']}")
    
    # タイムアウトコールバックの作成
    timeout_callback = TimeoutCallback(settings["timeout_seconds"])
    
    try:
        # 最適化実行
        logging.info(f"Starting optimization with timeout of {settings['timeout_seconds']} seconds")
        study.optimize(
            objective_with_args, 
            n_trials=settings["n_trials"],
            callbacks=[timeout_callback]
        )
    except KeyboardInterrupt:
        logging.warning("ユーザーによって実験が中断されました")
    except Exception as e:
        if timeout_flag:
            logging.warning("タイムアウトにより実験が中断されました")
        else:
            logging.error(f"予期せぬエラーが発生しました: {e}")
    finally:
        # 結果の保存
        save_results()
    
    return current_study


def parse_args():
    """コマンドライン引数の解析"""
    parser = argparse.ArgumentParser(description="Ackley 関数のベンチマーク実験")
    
    # 基本設定
    parser.add_argument("--timestamp", type=str, help="実験のタイムスタンプ")
    parser.add_argument("--seed", type=int, default=0, help="乱数シード")
    parser.add_argument("--n_dim", type=int, default=2, help="問題の次元数")
    parser.add_argument("--cat_num", type=int, default=5, help="カテゴリの数")
    parser.add_argument("--max_radius", type=float, default=2.0, 
                        help="制約の半径（大きい値=制約なし）")
    parser.add_argument("--n_trials", type=int, default=15, help="最適化の試行回数")
    parser.add_argument("--timeout", type=int, default=3600, help="タイムアウト時間（秒）。デフォルト：1時間(3600秒)")
    
    # サンプラー設定
    parser.add_argument("--sampler", type=str, choices=["random", "tpe", "tf_sdpa"], default="tf_sdpa",
                        help="使用するサンプラー")
    parser.add_argument("--n_startup_trials", type=int, default=1, 
                        help="ランダムサンプリングの初期試行数")
    
    # TFSdpaSampler の設定
    parser.add_argument("--mask_ratio", type=float, default=0.1, 
                        help="マスク比率")
    parser.add_argument("--include_observed_points", action="store_true",
                        help="観測済みの点を候補に含める")
    parser.add_argument("--tt_rank", type=int, default=2, 
                        help="テンソル分解のランク")
    parser.add_argument("--sdp_level", type=int, default=2, 
                        help="SDPレベル")
    
    # 出力設定
    parser.add_argument("--results_dir", type=str, help="結果を保存するディレクトリ")
    parser.add_argument("--plot_save_dir", type=str, help="プロットを保存するディレクトリ")
    
    return parser.parse_args()


def get_sampler(sampler_type: str, seed: int, args, constraint_tensor=None):
    """サンプラーを作成して返す"""
    if sampler_type == "random":
        return RandomSampler(seed=seed)
    elif sampler_type == "tpe":
        return TPESampler(seed=seed, n_startup_trials=args.n_startup_trials)
    elif sampler_type == "tf_sdpa":
        return TFSdpaSampler(
            seed=seed,
            sampler_params={
                "n_startup_trials": args.n_startup_trials,
                "mask_ratio": args.mask_ratio,
                "include_observed_points": args.include_observed_points,
            },
            tf_params={
                "rank": args.tt_rank,
                "sdp_level": args.sdp_level,
            },
            tensor_constraint=constraint_tensor
        )
    else:
        raise ValueError(f"Unsupported sampler type: {sampler_type}")


if __name__ == "__main__":
    # コマンドライン引数の解析
    args = parse_args()
    
    # タイムアウトシグナルハンドラを設定
    signal.signal(signal.SIGALRM, timeout_handler)
    # タイムアウト時間をセット（秒単位）
    signal.alarm(args.timeout)
    
    # タイムスタンプの設定
    if not args.timestamp:
        args.timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 結果ディレクトリの設定
    if not args.results_dir:
        args.results_dir = os.path.join("results", args.timestamp)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # プロット保存ディレクトリの設定（オプション）
    if args.plot_save_dir:
        os.makedirs(args.plot_save_dir, exist_ok=True)
    
    # ログファイル名の設定
    is_constrained = args.max_radius < (args.cat_num * 2)
    constraint_str = "constrained" if is_constrained else "unconstrained"
    log_filename_base = f"ackley_cat{args.cat_num}_rad{args.max_radius}_{constraint_str}_{args.sampler}_seed{args.seed}"
    
    # ロガーを設定
    log_filepath = set_logger(log_filename_base, args.results_dir)
    
    # SQLiteストレージを設定
    storage_filename = f"{log_filename_base}.db"
    storage_path = os.path.join(args.results_dir, storage_filename)
    storage_url = f"sqlite:///{storage_path}"
    
    # 制約テンソルを作成
    constraint_tensor, _ = create_constraint_tensor(args.cat_num, args.n_dim, args.max_radius)
    
    # サンプラーを取得
    sampler = get_sampler(args.sampler, args.seed, args, constraint_tensor)

    # 設定をまとめる
    settings = {
        "name": f"{args.timestamp}_{log_filename_base}",
        "log_filename_base": log_filename_base,
        "seed": args.seed,
        "cat_num": args.cat_num,
        "max_radius": args.max_radius,
        "n_dim": args.n_dim,
        "n_trials": args.n_trials,
        "timeout_seconds": args.timeout,
        "storage": storage_url,
        "results_dir": args.results_dir,
        "plot_save_dir": args.plot_save_dir,
        "sampler": sampler,
        "sampler_type": args.sampler,
    }
    
    # 設定ログを出力
    logging.info(f"Experiment settings: {settings}")
    
    try:
        # 最適化実行
        study = run_bo(settings)
        
        # 終了メッセージ
        if timeout_flag:
            logging.warning("タイムアウトにより最適化が中断されました")
            logging.info(f"Completed {len(study.trials)} trials before timeout")
        else:
            logging.info("最適化が正常に完了しました")
    except KeyboardInterrupt:
        logging.warning("ユーザーによって実験が中断されました")
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")
    finally:
        # タイムアウトアラームを解除
        signal.alarm(0)


