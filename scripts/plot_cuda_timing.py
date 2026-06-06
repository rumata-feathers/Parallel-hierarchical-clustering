#!/usr/bin/env python3
"""
Parses cuda_timing.log and plots copy vs compute breakdown per dataset.
"""
import re
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt


def parse_cuda_log(path):
    rows = []
    current = {}
    with open(path) as f:
        for line in f:
            m = re.match(r'\[cuda\] total copy:\s+([\d.]+) ms', line)
            if m:
                current['copy_ms'] = float(m.group(1))
            m = re.match(r'\[cuda\] total compute:\s+([\d.]+) ms', line)
            if m:
                current['compute_ms'] = float(m.group(1))
            m = re.match(r'\[cuda\] copy fraction:\s+([\d.]+)%', line)
            if m:
                current['copy_pct'] = float(m.group(1))
                rows.append(dict(current))
                current = {}
    return pd.DataFrame(rows)


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    log_path = os.path.join(results_dir, 'cuda', 'cuda_timing.log')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(log_path):
        print(f'No cuda timing log found at {log_path} — skipping.')
        sys.exit(0)

    df = parse_cuda_log(log_path)
    if df.empty:
        print('No cuda timing data found.')
        sys.exit(1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.bar(range(len(df)), df['copy_ms'], label='copy', color='steelblue')
    ax1.bar(range(len(df)), df['compute_ms'], bottom=df['copy_ms'],
            label='compute', color='darkorange')
    ax1.set_xlabel('Run index')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('CUDA: copy vs compute per run')
    ax1.legend()

    ax2.hist(df['copy_pct'], bins=20, color='steelblue', edgecolor='white')
    ax2.axvline(df['copy_pct'].mean(), color='red', linestyle='--',
                label=f"mean {df['copy_pct'].mean():.1f}%")
    ax2.set_xlabel('Copy fraction (%)')
    ax2.set_ylabel('Count')
    ax2.set_title('CUDA: distribution of copy fraction')
    ax2.legend()

    fig.tight_layout()
    out = os.path.join(plots_dir, 'cuda_copy_vs_compute.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved: {out}')
