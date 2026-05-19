# -*- coding: utf-8 -*-
import argparse
import csv
import math
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime


LINE_RE = re.compile(
    r"Epoch:\s*(?P<epoch>\S+)\s*\|\s*"
    r"AP @0\.3:\s*(?P<ap30>[0-9.]+)\s*\|\s*"
    r"AP @0\.5:\s*(?P<ap50>[0-9.]+)\s*\|\s*"
    r"AP @0\.7:\s*(?P<ap70>[0-9.]+)\s*\|\s*"
    r"comm_rate:\s*(?P<comm>[0-9.]+)"
    r"(?:\s*\|\s*comm_thre:\s*(?P<comm_thre>[0-9.]+))?"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically find comm_thre for a target comm_rate, then repeat inference N times."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint/log directory")
    parser.add_argument(
        "--fusion_method",
        type=str,
        default="intermediate_with_comm",
        choices=["intermediate_with_comm"],
        help="Only intermediate_with_comm is supported here.",
    )
    parser.add_argument("--eval_epoch", type=str, default=None, help="Checkpoint epoch to evaluate")
    parser.add_argument(
        "--target_comm_rate",
        type=float,
        required=True,
        help="Target communication rate. Use 0.6 or 60 for 60%%.",
    )
    parser.add_argument(
        "--init_comm_thre",
        type=float,
        default=1e-3,
        help="Initial guess for comm_thre. Example: 0.0011",
    )
    parser.add_argument(
        "--repeat_times",
        type=int,
        default=10,
        help="How many times to repeat inference after finding the threshold",
    )
    parser.add_argument("--save_vis_n", type=int, default=0, help="Visualization count per run; suggest 0")
    parser.add_argument("--save_npy", action="store_true", help="Whether to save npy results")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python executable")

    parser.add_argument(
        "--expand_factor",
        type=float,
        default=3.0,
        help="Multiplicative factor for automatic bracket expansion",
    )
    parser.add_argument(
        "--min_comm_thre",
        type=float,
        default=1e-8,
        help="Smallest threshold allowed during auto-expansion",
    )
    parser.add_argument(
        "--max_comm_thre",
        type=float,
        default=1.0,
        help="Largest threshold allowed during auto-expansion",
    )
    parser.add_argument(
        "--coarse_steps",
        type=int,
        default=8,
        help="Number of log-spaced coarse search points inside the found bracket",
    )
    parser.add_argument(
        "--refine_iters",
        type=int,
        default=12,
        help="Binary refinement iterations after coarse search",
    )
    parser.add_argument(
        "--target_tol",
        type=float,
        default=0.005,
        help="Early-stop tolerance on absolute comm_rate error",
    )

    parser.add_argument("--show_child_output", action="store_true", help="Show raw child inference output")
    parser.add_argument("--output_prefix", type=str, default=None, help="Output prefix; default: auto-generated")
    parser.add_argument(
        "--append_result_txt",
        action="store_true",
        help="Append only final summary to model_dir/result_repeat.txt",
    )
    return parser.parse_args()


def normalize_target_rate(x: float) -> float:
    return x / 100.0 if x > 1.0 else x


def parse_last_metric_line(text: str):
    last = None
    for m in LINE_RE.finditer(text):
        last = m
    if last is None:
        return None
    gd = last.groupdict()
    return {
        "epoch": gd["epoch"],
        "ap30": float(gd["ap30"]),
        "ap50": float(gd["ap50"]),
        "ap70": float(gd["ap70"]),
        "comm_rate": float(gd["comm"]),
        "comm_thre": None if gd.get("comm_thre") is None else float(gd["comm_thre"]),
    }


def snapshot_result_txt(model_dir):
    path = os.path.join(model_dir, "result.txt")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return True, f.read()
    return False, b""


def restore_result_txt(model_dir, existed_before, content_before):
    path = os.path.join(model_dir, "result.txt")
    if existed_before:
        with open(path, "wb") as f:
            f.write(content_before)
    else:
        if os.path.exists(path):
            os.remove(path)


def logspace(min_val, max_val, steps):
    if steps <= 1:
        return [min_val]
    a = math.log10(min_val)
    b = math.log10(max_val)
    return [10 ** (a + (b - a) * i / (steps - 1)) for i in range(steps)]


def geometric_mid(lo, hi):
    return math.sqrt(lo * hi)


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


class InferenceRunner:
    def __init__(self, args, use_cache=True):
        self.args = args
        self.use_cache = use_cache
        self.cache = {}

    def _cache_key(self, comm_thre):
        return f"{comm_thre:.16g}"

    def run(self, comm_thre):
        key = self._cache_key(comm_thre)

        if self.use_cache and key in self.cache:
            return self.cache[key]

        cmd = [
            self.args.python_exec,
            "-m",
            "opencood.tools.inference",
            "--model_dir",
            self.args.model_dir,
            "--fusion_method",
            self.args.fusion_method,
            "--save_vis_n",
            str(self.args.save_vis_n),
            "--comm_thre",
            str(comm_thre),
        ]
        if self.args.eval_epoch is not None:
            cmd += ["--eval_epoch", str(self.args.eval_epoch)]
        if self.args.save_npy:
            cmd += ["--save_npy"]

        existed_before, content_before = snapshot_result_txt(self.args.model_dir)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        restore_result_txt(self.args.model_dir, existed_before, content_before)

        if self.args.show_child_output:
            print(proc.stdout)

        if proc.returncode != 0:
            raise RuntimeError(
                f"Inference failed with exit code {proc.returncode} at comm_thre={comm_thre}.\n"
                f"Last output:\n{proc.stdout[-4000:]}"
            )

        metrics = parse_last_metric_line(proc.stdout)
        if metrics is None:
            raise RuntimeError(
                f"Inference finished but no metric line was parsed at comm_thre={comm_thre}.\n"
                f"Last output:\n{proc.stdout[-4000:]}"
            )

        if self.use_cache:
            self.cache[key] = metrics
        return metrics


def row_from_metrics(phase, idx, comm_thre, metrics):
    return {
        "phase": phase,
        "idx": idx,
        "comm_thre": comm_thre,
        "epoch": metrics["epoch"],
        "ap30": metrics["ap30"],
        "ap50": metrics["ap50"],
        "ap70": metrics["ap70"],
        "comm_rate": metrics["comm_rate"],
    }


def best_row(rows, target):
    best = None
    best_err = None
    for r in rows:
        err = abs(r["comm_rate"] - target)
        if best is None or err < best_err:
            best = r
            best_err = err
    return best, best_err


def main():
    args = parse_args()
    target = normalize_target_rate(args.target_comm_rate)

    # 搜索阶段允许缓存
    runner = InferenceRunner(args, use_cache=True)

    print(f"Target comm_rate = {target:.6f}")
    print(f"Initial comm_thre guess = {args.init_comm_thre:.8f}")

    search_rows = []

    init_metrics = runner.run(args.init_comm_thre)
    init_row = row_from_metrics("init", 1, args.init_comm_thre, init_metrics)
    search_rows.append(init_row)

    print(
        f"Init | comm_thre={args.init_comm_thre:.8f} | "
        f"comm_rate={init_metrics['comm_rate']:.6f} | AP@0.5={init_metrics['ap50']:.4f}"
    )

    best, best_err = best_row(search_rows, target)
    if best_err <= args.target_tol:
        print("\nInitial guess already close enough to target.")
        selected_thre = best["comm_thre"]
    else:
        bracket_low = None
        bracket_high = None

        current_thre = args.init_comm_thre
        current_rate = init_metrics["comm_rate"]

        if current_rate > target:
            prev_row = init_row
            step = 0
            while True:
                step += 1
                next_thre = current_thre * args.expand_factor
                if next_thre > args.max_comm_thre:
                    break
                next_metrics = runner.run(next_thre)
                next_row = row_from_metrics("expand_up", step, next_thre, next_metrics)
                search_rows.append(next_row)
                print(
                    f"ExpandUp {step:02d} | comm_thre={next_thre:.8f} | "
                    f"comm_rate={next_metrics['comm_rate']:.6f} | AP@0.5={next_metrics['ap50']:.4f}"
                )

                candidate_best, candidate_err = best_row([best, next_row], target)
                if candidate_err < best_err:
                    best, best_err = candidate_best, candidate_err

                if next_metrics["comm_rate"] <= target:
                    bracket_low = prev_row
                    bracket_high = next_row
                    break

                prev_row = next_row
                current_thre = next_thre

        else:
            prev_row = init_row
            step = 0
            while True:
                step += 1
                next_thre = current_thre / args.expand_factor
                if next_thre < args.min_comm_thre:
                    break
                next_metrics = runner.run(next_thre)
                next_row = row_from_metrics("expand_down", step, next_thre, next_metrics)
                search_rows.append(next_row)
                print(
                    f"ExpandDn {step:02d} | comm_thre={next_thre:.8f} | "
                    f"comm_rate={next_metrics['comm_rate']:.6f} | AP@0.5={next_metrics['ap50']:.4f}"
                )

                candidate_best, candidate_err = best_row([best, next_row], target)
                if candidate_err < best_err:
                    best, best_err = candidate_best, candidate_err

                if next_metrics["comm_rate"] >= target:
                    bracket_low = next_row
                    bracket_high = prev_row
                    break

                prev_row = next_row
                current_thre = next_thre

        if bracket_low is None or bracket_high is None:
            print("\nCould not bracket the target within allowed threshold bounds.")
            print("Will use the closest observed threshold directly.")
            selected_thre = best["comm_thre"]
        else:
            print("\nBracket found:")
            print(
                f"low_thre={bracket_low['comm_thre']:.8f}, low_comm_rate={bracket_low['comm_rate']:.6f}"
            )
            print(
                f"high_thre={bracket_high['comm_thre']:.8f}, high_comm_rate={bracket_high['comm_rate']:.6f}"
            )

            lo = bracket_low["comm_thre"]
            hi = bracket_high["comm_thre"]
            coarse_points = logspace(lo, hi, args.coarse_steps)

            print("\n========== Coarse search inside bracket ==========")
            for idx, thre in enumerate(coarse_points, start=1):
                metrics = runner.run(thre)
                row = row_from_metrics("coarse", idx, thre, metrics)
                search_rows.append(row)
                print(
                    f"Coarse {idx:02d}/{len(coarse_points):02d} | "
                    f"comm_thre={thre:.8f} | comm_rate={metrics['comm_rate']:.6f} | "
                    f"AP@0.5={metrics['ap50']:.4f}"
                )
                err = abs(metrics["comm_rate"] - target)
                if err < best_err:
                    best = row
                    best_err = err

            low_row = bracket_low
            high_row = bracket_high

            print("\n========== Local binary refinement ==========")
            for it in range(1, args.refine_iters + 1):
                mid_thre = geometric_mid(low_row["comm_thre"], high_row["comm_thre"])
                metrics = runner.run(mid_thre)
                row = row_from_metrics("refine", it, mid_thre, metrics)
                search_rows.append(row)

                print(
                    f"Refine {it:02d}/{args.refine_iters:02d} | "
                    f"comm_thre={mid_thre:.8f} | comm_rate={metrics['comm_rate']:.6f} | "
                    f"AP@0.5={metrics['ap50']:.4f}"
                )

                err = abs(metrics["comm_rate"] - target)
                if err < best_err:
                    best = row
                    best_err = err

                if err <= args.target_tol:
                    print("Reached target tolerance early.")
                    break

                if metrics["comm_rate"] > target:
                    low_row = row
                else:
                    high_row = row

            selected_thre = best["comm_thre"]

    print("\n========== Selected threshold ==========")
    print(
        f"selected_comm_thre={selected_thre:.8f} | "
        f"closest_comm_rate={best['comm_rate']:.6f} | "
        f"target_comm_rate={target:.6f} | "
        f"abs_err={abs(best['comm_rate'] - target):.6f}"
    )

    print("\n========== Repeat final inference ==========")
    repeat_rows = []

    # 关键修复：repeat 阶段禁止缓存，确保每次都真跑
    repeat_runner = InferenceRunner(args, use_cache=False)

    for run_idx in range(1, args.repeat_times + 1):
        metrics = repeat_runner.run(selected_thre)
        metrics["run"] = run_idx
        repeat_rows.append(metrics)
        print(
            f"Run {run_idx:02d}/{args.repeat_times:02d} | "
            f"epoch={metrics['epoch']} | "
            f"AP@0.3={metrics['ap30']:.4f} | "
            f"AP@0.5={metrics['ap50']:.4f} | "
            f"AP@0.7={metrics['ap70']:.4f} | "
            f"comm_rate={metrics['comm_rate']:.6f} | "
            f"comm_thre={selected_thre:.8f}"
        )

    ap30s = [r["ap30"] for r in repeat_rows]
    ap50s = [r["ap50"] for r in repeat_rows]
    ap70s = [r["ap70"] for r in repeat_rows]
    comms = [r["comm_rate"] for r in repeat_rows]

    ap30_mean, ap30_std = mean_std(ap30s)
    ap50_mean, ap50_std = mean_std(ap50s)
    ap70_mean, ap70_std = mean_std(ap70s)
    comm_mean, comm_std = mean_std(comms)

    print("\n========== Final summary ==========")
    print(
        f"selected_comm_thre={selected_thre:.8f} | target_comm_rate={target:.6f}\n"
        f"AP@0.3={ap30_mean:.4f} ± {ap30_std:.4f} | "
        f"AP@0.5={ap50_mean:.4f} ± {ap50_std:.4f} | "
        f"AP@0.7={ap70_mean:.4f} ± {ap70_std:.4f} | "
        f"comm_rate={comm_mean:.6f} ± {comm_std:.6f}"
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_prefix is None:
        output_prefix = os.path.join(args.model_dir, f"auto_comm_repeat_{ts}")
    else:
        output_prefix = args.output_prefix

    search_csv = output_prefix + "_search.csv"
    repeat_csv = output_prefix + "_repeat.csv"
    summary_txt = output_prefix + "_summary.txt"

    with open(search_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["phase", "idx", "comm_thre", "epoch", "ap30", "ap50", "ap70", "comm_rate"],
        )
        writer.writeheader()
        for row in search_rows:
            writer.writerow(row)

    with open(repeat_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "epoch", "ap30", "ap50", "ap70", "comm_rate", "comm_thre"],
        )
        writer.writeheader()
        for row in repeat_rows:
            writer.writerow({
                "run": row["run"],
                "epoch": row["epoch"],
                "ap30": row["ap30"],
                "ap50": row["ap50"],
                "ap70": row["ap70"],
                "comm_rate": row["comm_rate"],
                "comm_thre": selected_thre,
            })
        writer.writerow({
            "run": "mean",
            "epoch": repeat_rows[0]["epoch"],
            "ap30": f"{ap30_mean:.6f}",
            "ap50": f"{ap50_mean:.6f}",
            "ap70": f"{ap70_mean:.6f}",
            "comm_rate": f"{comm_mean:.6f}",
            "comm_thre": f"{selected_thre:.8f}",
        })
        writer.writerow({
            "run": "std",
            "epoch": repeat_rows[0]["epoch"],
            "ap30": f"{ap30_std:.6f}",
            "ap50": f"{ap50_std:.6f}",
            "ap70": f"{ap70_std:.6f}",
            "comm_rate": f"{comm_std:.6f}",
            "comm_thre": f"{selected_thre:.8f}",
        })

    with open(summary_txt, "w") as f:
        f.write(f"target_comm_rate={target:.6f}\n")
        f.write(f"selected_comm_thre={selected_thre:.8f}\n")
        f.write(f"repeat_times={args.repeat_times}\n")
        f.write(f"AP@0.3={ap30_mean:.4f} ± {ap30_std:.4f}\n")
        f.write(f"AP@0.5={ap50_mean:.4f} ± {ap50_std:.4f}\n")
        f.write(f"AP@0.7={ap70_mean:.4f} ± {ap70_std:.4f}\n")
        f.write(f"comm_rate={comm_mean:.6f} ± {comm_std:.6f}\n")

    print(f"Saved search CSV to: {search_csv}")
    print(f"Saved repeat CSV to: {repeat_csv}")
    print(f"Saved summary TXT to: {summary_txt}")

    if args.append_result_txt:
        out_txt = os.path.join(args.model_dir, "result_repeat.txt")
        with open(out_txt, "a+") as f:
            msg = (
                f"target_comm_rate={target:.6f} | "
                f"selected_comm_thre={selected_thre:.8f} | "
                f"repeat_times={args.repeat_times} | "
                f"eval_epoch={args.eval_epoch} | "
                f"AP@0.3={ap30_mean:.4f}±{ap30_std:.4f} | "
                f"AP@0.5={ap50_mean:.4f}±{ap50_std:.4f} | "
                f"AP@0.7={ap70_mean:.4f}±{ap70_std:.4f} | "
                f"comm_rate={comm_mean:.6f}±{comm_std:.6f}"
            )
            f.write(msg + "\n")
        print(f"Appended final summary to: {out_txt}")


if __name__ == "__main__":
    main()