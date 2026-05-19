# -*- coding: utf-8 -*-
import argparse
import csv
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
        description="Repeat inference N times, suppress child tqdm logs, and summarize results."
    )
    parser.add_argument("--model_dir", type=str, required=True, help="Checkpoint/log directory")
    parser.add_argument(
        "--fusion_method",
        type=str,
        required=True,
        choices=["late", "early", "intermediate", "intermediate_with_comm", "no"],
    )
    parser.add_argument("--repeat_times", type=int, default=5, help="How many times to repeat inference")
    parser.add_argument("--eval_epoch", type=str, default=None, help="Checkpoint epoch to evaluate")
    parser.add_argument("--comm_thre", type=float, default=None, help="Communication threshold")
    parser.add_argument("--save_vis_n", type=int, default=0, help="Visualization count per run; suggest 0")
    parser.add_argument("--save_npy", action="store_true", help="Whether to save npy results")
    parser.add_argument("--python_exec", type=str, default=sys.executable, help="Python executable")
    parser.add_argument("--output_csv", type=str, default=None, help="Optional csv output path")
    parser.add_argument(
        "--append_result_txt",
        action="store_true",
        help="Append only the final summarized stats to model_dir/result_repeat.txt",
    )
    parser.add_argument(
        "--show_child_output",
        action="store_true",
        help="Show raw output from each child inference process",
    )
    return parser.parse_args()


def build_cmd(args):
    cmd = [
        args.python_exec,
        "-m",
        "opencood.tools.inference",
        "--model_dir",
        args.model_dir,
        "--fusion_method",
        args.fusion_method,
        "--save_vis_n",
        str(args.save_vis_n),
    ]
    if args.eval_epoch is not None:
        cmd += ["--eval_epoch", str(args.eval_epoch)]
    if args.comm_thre is not None:
        cmd += ["--comm_thre", str(args.comm_thre)]
    if args.save_npy:
        cmd += ["--save_npy"]
    return cmd


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


def mean_std(values):
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


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


def main():
    args = parse_args()
    cmd = build_cmd(args)

    rows = []
    for run_idx in range(1, args.repeat_times + 1):
        existed_before, content_before = snapshot_result_txt(args.model_dir)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # restore result.txt so repeated runs don't pollute it
        restore_result_txt(args.model_dir, existed_before, content_before)

        if args.show_child_output:
            print(proc.stdout)

        metrics = parse_last_metric_line(proc.stdout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Run {run_idx} failed with exit code {proc.returncode}.\n"
                f"Last output:\n{proc.stdout[-4000:]}"
            )
        if metrics is None:
            raise RuntimeError(
                f"Run {run_idx} finished but no metric line was parsed.\n"
                f"Last output:\n{proc.stdout[-4000:]}"
            )

        metrics["run"] = run_idx
        rows.append(metrics)

        extra = f" | comm_thre={metrics['comm_thre']:.4f}" if metrics["comm_thre"] is not None else ""
        print(
            f"Run {run_idx:02d}/{args.repeat_times:02d} | epoch={metrics['epoch']} | "
            f"AP@0.3={metrics['ap30']:.4f} | AP@0.5={metrics['ap50']:.4f} | "
            f"AP@0.7={metrics['ap70']:.4f} | comm_rate={metrics['comm_rate']:.6f}{extra}"
        )

    ap30s = [r["ap30"] for r in rows]
    ap50s = [r["ap50"] for r in rows]
    ap70s = [r["ap70"] for r in rows]
    comms = [r["comm_rate"] for r in rows]

    ap30_mean, ap30_std = mean_std(ap30s)
    ap50_mean, ap50_std = mean_std(ap50s)
    ap70_mean, ap70_std = mean_std(ap70s)
    comm_mean, comm_std = mean_std(comms)

    print("\n========== Summary ==========")
    print(
        f"Mean ± Std | "
        f"AP@0.3={ap30_mean:.4f} ± {ap30_std:.4f} | "
        f"AP@0.5={ap50_mean:.4f} ± {ap50_std:.4f} | "
        f"AP@0.7={ap70_mean:.4f} ± {ap70_std:.4f} | "
        f"comm_rate={comm_mean:.6f} ± {comm_std:.6f}"
    )

    output_csv = args.output_csv
    if output_csv is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = os.path.join(
            args.model_dir,
            f"repeat_inference_{args.fusion_method}_{ts}.csv",
        )

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "epoch", "ap30", "ap50", "ap70", "comm_rate", "comm_thre"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        writer.writerow({
            "run": "mean",
            "epoch": rows[0]["epoch"],
            "ap30": f"{ap30_mean:.6f}",
            "ap50": f"{ap50_mean:.6f}",
            "ap70": f"{ap70_mean:.6f}",
            "comm_rate": f"{comm_mean:.6f}",
            "comm_thre": rows[0]["comm_thre"] if rows[0]["comm_thre"] is not None else "",
        })
        writer.writerow({
            "run": "std",
            "epoch": rows[0]["epoch"],
            "ap30": f"{ap30_std:.6f}",
            "ap50": f"{ap50_std:.6f}",
            "ap70": f"{ap70_std:.6f}",
            "comm_rate": f"{comm_std:.6f}",
            "comm_thre": rows[0]["comm_thre"] if rows[0]["comm_thre"] is not None else "",
        })

    print(f"Saved CSV summary to: {output_csv}")

    if args.append_result_txt:
        out_txt = os.path.join(args.model_dir, "result_repeat.txt")
        with open(out_txt, "a+") as f:
            msg = (
                f"repeat_times={args.repeat_times} | fusion_method={args.fusion_method} | "
                f"eval_epoch={args.eval_epoch} | "
                f"AP@0.3={ap30_mean:.4f}±{ap30_std:.4f} | "
                f"AP@0.5={ap50_mean:.4f}±{ap50_std:.4f} | "
                f"AP@0.7={ap70_mean:.4f}±{ap70_std:.4f} | "
                f"comm_rate={comm_mean:.6f}±{comm_std:.6f}"
            )
            if rows[0]["comm_thre"] is not None:
                msg += f" | comm_thre={rows[0]['comm_thre']:.4f}"
            f.write(msg + "\n")
        print(f"Appended summary to: {out_txt}")


if __name__ == "__main__":
    main()