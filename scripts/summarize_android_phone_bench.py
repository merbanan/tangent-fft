#!/usr/bin/env python3
"""Summarize repeated output from run_android_phone_bench.sh."""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path


ALGORITHMS = (
    "lane2-neon",
    "lane4-neon-fused",
    "ffmpeg-neon-natural",
    "ffmpeg-neon-prepermuted",
)


def read_results(directory: Path):
    values = defaultdict(list)
    for path in sorted(directory.glob("*.txt")):
        cluster = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("META_cluster="):
                cluster = line.partition("=")[2]
                continue
            fields = line.split()
            if (
                cluster is not None
                and len(fields) == 5
                and fields[0].isdigit()
                and fields[1] in ALGORITHMS
            ):
                n = int(fields[0])
                values[(cluster, n, fields[1])].append(int(fields[2]))
    return values


def median_integer(samples):
    return int(statistics.median(samples))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-runs", type=int, default=5)
    args = parser.parse_args()
    values = read_results(args.directory)

    for cluster in ("a55", "a78"):
        sizes = sorted(
            n for seen_cluster, n, algorithm in values
            if seen_cluster == cluster and algorithm == ALGORITHMS[0]
        )
        print(f"{cluster.upper()} (outer median, ns/FFT)")
        print(
            "N lane2 lane4 ffmpeg-natural ffmpeg-pre "
            "lane2/natural lane4/natural lane4/pre"
        )
        worst_spread = (-1.0, 0, "")
        for n in sizes:
            for algorithm in ALGORITHMS:
                sample_count = len(values[(cluster, n, algorithm)])
                if sample_count != args.expected_runs:
                    raise SystemExit(
                        f"{cluster} N={n} {algorithm}: expected "
                        f"{args.expected_runs} runs, found {sample_count}"
                    )
                samples = values[(cluster, n, algorithm)]
                median = median_integer(samples)
                spread = 100.0 * (max(samples) - min(samples)) / median
                worst_spread = max(
                    worst_spread, (spread, n, algorithm)
                )
            row = {
                algorithm: median_integer(values[(cluster, n, algorithm)])
                for algorithm in ALGORITHMS
            }
            natural = row["ffmpeg-neon-natural"]
            pre = row["ffmpeg-neon-prepermuted"]
            print(
                f"{n} {row['lane2-neon']} {row['lane4-neon-fused']} "
                f"{natural} {pre} "
                f"{row['lane2-neon'] / natural:.3f} "
                f"{row['lane4-neon-fused'] / natural:.3f} "
                f"{row['lane4-neon-fused'] / pre:.3f}"
            )
        print(
            "worst_outer_spread_percent "
            f"{worst_spread[0]:.3f} N={worst_spread[1]} "
            f"{worst_spread[2]}"
        )
        print()


if __name__ == "__main__":
    main()
