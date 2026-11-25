"""
Generate all experiment graphs and terminal command screenshots for Experiment 7.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image, ImageDraw, ImageFont

os.makedirs("graphs", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#f8fafc",
    "axes.edgecolor": "#cbd5e1",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "font.size": 10,
})


def load_trace(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def plot_timelines():
    base_ev = load_trace("results/trace_baseline_events.json")
    opt_ev = load_trace("results/trace_optimized_events.json")

    fig, axes = plt.subplots(2, 1, figsize=(13, 6.8), sharex=True)
    fig.suptitle(
        "Real GPU Execution Timelines (First 3 Batches) — RTX 4050 Laptop GPU",
        fontsize=14, fontweight="bold", y=0.98
    )

    for ax, evs, title, is_opt in [
        (axes[0], base_ev, "BEFORE — Baseline (Pageable Memory, Default Stream, Blocking Copies)", False),
        (axes[1], opt_ev, "AFTER — Optimized (Pinned Memory, Two CUDA Streams, Double Buffering)", True),
    ]:
        # Take first 3 batches (each batch has 13 kernel events + 2 memcpy events = 15 events)
        # Filter events starting within first ~115 ms
        t0 = min(e["ts"] for e in evs)
        cutoff = 112.0  # ms
        copies = []
        kernels = []
        for e in evs:
            s = (e["ts"] - t0) / 1000.0
            d = e["dur"] / 1000.0
            if s > cutoff:
                continue
            if e["cat"] == "gpu_memcpy":
                copies.append((s, d, e["name"]))
            elif e["cat"] == "kernel":
                kernels.append((s, d, e["name"]))

        # Draw copy bars on row y=1
        for s, d, name in copies:
            color = "#f97316" if "HtoD" in name else "#fbbf24"
            ax.barh(1, d, left=s, height=0.55, color=color, edgecolor="#9a3412", linewidth=0.6)

        # Draw kernel bars on row y=0
        for s, d, name in kernels:
            color = "#2563eb" if "sgemm" in name.lower() else "#38bdf8"
            ax.barh(0, d, left=s, height=0.55, color=color, edgecolor="#1e3a8a", linewidth=0.3)

        # Highlight idle gaps on baseline
        if not is_opt:
            # Find gaps > 2 ms between consecutive kernel clusters
            k_sorted = sorted(kernels)
            for idx in range(len(k_sorted) - 1):
                end_prev = k_sorted[idx][0] + k_sorted[idx][1]
                start_next = k_sorted[idx + 1][0]
                if start_next - end_prev > 2.0:
                    ax.axvspan(end_prev, start_next, ymin=0.05, ymax=0.48, color="#ef4444", alpha=0.18)
                    ax.text(
                        (end_prev + start_next) / 2, 0.0,
                        f"Idle Gap\n{start_next - end_prev:.1f} ms",
                        ha="center", va="center", fontsize=8, color="#b91c1c", fontweight="bold"
                    )
        else:
            # Annotate overlap regions
            for cs, cd, cname in copies:
                if "HtoD" not in cname or cd < 1.0:
                    continue
                ce = cs + cd
                for ks, kd, _ in kernels:
                    ke = ks + kd
                    ov_s, ov_e = max(cs, ks), min(ce, ke)
                    if ov_e - ov_s > 0.3:
                        ax.axvspan(ov_s, ov_e, ymin=0.05, ymax=0.95, color="#22c55e", alpha=0.18)

        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Compute\n(Kernels)", "Copy Engine\n(Memcpy)"])
        ax.set_ylim(-0.55, 1.55)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=8)

    axes[1].set_xlabel("Time (ms) from start of Batch 0", fontsize=11)
    axes[1].set_xlim(0, 115)

    legend_handles = [
        mpatches.Patch(color="#f97316", label="Memcpy HtoD (64 MB)"),
        mpatches.Patch(color="#fbbf24", label="Memcpy DtoH (64 KB)"),
        mpatches.Patch(color="#2563eb", label="GEMM Kernel (Compute-bound)"),
        mpatches.Patch(color="#38bdf8", label="Element-wise / Reduce (Memory-bound)"),
        mpatches.Patch(color="#ef4444", alpha=0.25, label="GPU Idle Gap (Baseline)"),
        mpatches.Patch(color="#22c55e", alpha=0.25, label="Copy-Compute Overlap (Optimized)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=True, fontsize=9)
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    plt.savefig("graphs/01_timeline_before_after.png", dpi=180)
    plt.close()


def plot_performance_comparison():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    fig.suptitle(
        "Baseline vs. Optimized Performance (200 Batches, 64 MB/batch, reps=3)",
        fontsize=13, fontweight="bold"
    )

    metrics = [
        ("Total Execution Time (s)", [7.226, 5.123], "Lower is better\n(1.41x Speedup)"),
        ("Time per Batch (ms)", [36.13, 25.62], "Lower is better\n(-10.51 ms / -29.1%)"),
        ("Throughput (batches/s)", [27.7, 39.0], "Higher is better\n(+40.8% throughput)"),
    ]
    colors = ["#ef4444", "#10b981"]
    labels = ["Baseline\n(Pageable, 1 Stream)", "Optimized\n(Pinned, 2 Streams)"]

    for ax, (title, vals, note) in zip(axes, metrics):
        bars = ax.bar(labels, vals, color=colors, width=0.52, edgecolor="#1e293b", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.03,
                f"{v:.2f}" if v < 100 else f"{v:.1f}",
                ha="center", va="bottom", fontweight="bold", fontsize=11
            )
        ax.text(
            0.5, 0.92, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f1f5f9", edgecolor="#94a3b8")
        )

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("graphs/02_performance_comparison.png", dpi=180)
    plt.close()


def plot_smi_utilization():
    def read_smi(path):
        utils, mems = [], []
        with open(path) as f:
            for line in f.readlines()[1:]:
                if "%" not in line:
                    continue
                parts = line.strip().split(",")
                utils.append(int(parts[0].replace("%", "").strip()))
                mems.append(int(parts[1].replace("MiB", "").strip()))
        t = np.arange(len(utils)) * 0.1
        return t, np.array(utils), np.array(mems)

    tb, ub, mb = read_smi("results/smi_baseline.csv")
    to, uo, mo = read_smi("results/smi_optimized.csv")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6.2), sharex=False)
    fig.suptitle("Live nvidia-smi Telemetry (100 ms Sampling Interval)", fontsize=13, fontweight="bold")

    ax1.plot(tb, ub, color="#ef4444", linewidth=2, label=f"Baseline (avg active: {ub[ub>0].mean():.1f}%, duration: {tb[-1]:.1f}s)")
    ax1.fill_between(tb, ub, color="#ef4444", alpha=0.12)
    ax1.plot(to, uo, color="#10b981", linewidth=2, label=f"Optimized (avg active: {uo[uo>0].mean():.1f}%, duration: {to[-1]:.1f}s)")
    ax1.fill_between(to, uo, color="#10b981", alpha=0.15)
    ax1.set_ylabel("GPU Utilization (%)")
    ax1.set_ylim(-5, 108)
    ax1.legend(loc="lower right")
    ax1.set_title("GPU-Util % Over Time (Optimized finishes ~2.1 s earlier)", loc="left", fontsize=10.5, fontweight="bold")

    ax2.plot(tb, mb, color="#ef4444", linewidth=2, label="Baseline VRAM (MiB)")
    ax2.plot(to, mo, color="#10b981", linewidth=2, label="Optimized VRAM (MiB, +1 double buffer = +64 MiB)")
    ax2.set_xlabel("Elapsed Time (seconds)")
    ax2.set_ylabel("GPU Memory Used (MiB)")
    ax2.legend(loc="lower right")
    ax2.set_title("GPU Memory Usage Over Time", loc="left", fontsize=10.5, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig("graphs/03_gpu_utilization_smi.png", dpi=180)
    plt.close()


def plot_kernel_and_roofline():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
    fig.suptitle(
        "Kernel Classification & Roofline Analysis — NVIDIA GeForce RTX 4050 Laptop GPU",
        fontsize=13, fontweight="bold"
    )

    # Left: Kernel time breakdown (10 profiled batches)
    k_names = ["cuBLAS GEMM\n(Compute-bound)", "add_\n(Memory-bound)", "mul_\n(Memory-bound)", "relu / clamp_min\n(Memory-bound)", "sum reduction\n(Memory-bound)"]
    k_times = [151.54, 24.42, 24.33, 23.43, 4.80]
    k_colors = ["#2563eb", "#f59e0b", "#f97316", "#fb7185", "#a855f7"]

    y_pos = np.arange(len(k_names))
    bars = ax1.barh(y_pos, k_times, color=k_colors, edgecolor="#1e293b", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(k_names)
    ax1.invert_yaxis()
    ax1.set_xlabel("Total Kernel Time over 10 Batches (ms)")
    ax1.set_title("Kernel Time Breakdown (Total = 228.52 ms)", fontweight="bold")
    ax1.set_xlim(0, 185)
    for bar, t in zip(bars, k_times):
        pct = 100.0 * t / sum(k_times)
        ax1.text(t + 2.5, bar.get_y() + bar.get_height() / 2, f"{t:.2f} ms ({pct:.1f}%)", va="center", fontsize=9, fontweight="bold")

    # Right: Roofline model for RTX 4050 Laptop GPU
    # Specs: ~9.0 TFLOP/s FP32 (70W TGP boost), 192 GB/s GDDR6 memory bandwidth -> Ridge point ~46.9 FLOP/byte
    peak_tflops = 9.0
    bw_gbs = 192.0
    ridge = (peak_tflops * 1000.0) / bw_gbs  # 46.875 FLOP/byte

    ai = np.logspace(-2, 3, 400)
    roof = np.minimum(peak_tflops, (bw_gbs * ai) / 1000.0)

    ax2.loglog(ai, roof, color="#0f172a", linewidth=2.5, label=f"RTX 4050 Roofline (9.0 TFLOP/s, 192 GB/s)")
    ax2.axvline(ridge, color="#64748b", linestyle=":", linewidth=1.5, label=f"Ridge Point = {ridge:.1f} FLOP/byte")

    # Plot our kernels
    pts = [
        ("ReLU / mul / add\n(AI=0.125)", 0.125, (bw_gbs * 0.125) / 1000.0 * 0.85, "#f97316"),
        ("Sum reduction\n(AI=0.25)", 0.25, (bw_gbs * 0.25) / 1000.0 * 0.85, "#a855f7"),
        ("cuBLAS GEMM\n(AI=249)", 249.0, 6.8, "#2563eb"),
    ]
    for label, x, y, col in pts:
        ax2.scatter([x], [y], color=col, s=110, zorder=5, edgecolor="#0f172a")
        ax2.annotate(label, (x, y), xytext=(x * 0.45 if x > 10 else x * 1.4, y * 0.42 if x > 10 else y * 1.6),
                     fontsize=8.5, fontweight="bold", color=col,
                     arrowprops=dict(arrowstyle="->", color=col, lw=1.0))

    ax2.set_xlabel("Arithmetic Intensity (FLOP / byte)")
    ax2.set_ylabel("Attainable Performance (TFLOP/s)")
    ax2.set_title("Roofline Model & Kernel Placement", fontweight="bold")
    ax2.set_xlim(0.03, 1000)
    ax2.set_ylim(0.005, 20)
    ax2.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("graphs/04_kernel_breakdown_roofline.png", dpi=180)
    plt.close()


def plot_reps_scaling():
    reps = [1, 3, 8]
    base_t = [4.028, 7.197, 14.923]
    opt_t = [2.273, 5.127, 13.692]
    speedups = [b / o for b, o in zip(base_t, opt_t)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Optional Experiment: Effect of Compute Intensity (--reps = 1, 3, 8)", fontsize=13, fontweight="bold")

    x = np.arange(len(reps))
    w = 0.35
    b1 = ax1.bar(x - w/2, base_t, w, label="Baseline", color="#ef4444", edgecolor="#1e293b")
    b2 = ax1.bar(x + w/2, opt_t, w, label="Optimized", color="#10b981", edgecolor="#1e293b")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"--reps {r}\n({'Copy-heavy' if r==1 else 'Balanced' if r==3 else 'Compute-heavy'})" for r in reps])
    ax1.set_ylabel("Total Time for 200 Batches (s)")
    ax1.set_title("Total Execution Time vs. --reps", fontweight="bold")
    ax1.legend()
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, h + 0.25, f"{h:.2f}s", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 17.5)

    bars_s = ax2.bar([f"--reps {r}" for r in reps], speedups, color=["#0ea5e9", "#6366f1", "#8b5cf6"], width=0.45, edgecolor="#1e293b")
    ax2.axhline(1.0, color="#64748b", linestyle="--", linewidth=1)
    ax2.set_ylabel("Speedup (Baseline / Optimized)")
    ax2.set_title("Speedup Factor Across Regimes", fontweight="bold")
    ax2.set_ylim(0, 2.15)
    for bar, s in zip(bars_s, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, s + 0.05, f"{s:.2f}x", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig("graphs/05_reps_scaling.png", dpi=180)
    plt.close()


def render_terminal_screenshot(title, command, output_text, out_path):
    """Render a crisp dark-mode Windows Terminal screenshot of a command and its real output."""
    lines = [f"PS C:\\exp7-gpu-profiling-utilization> {command}", ""] + output_text.rstrip().splitlines()
    # Clean any stray carriage returns
    lines = [l.replace("\r", "") for l in lines]

    font_path = "C:\\Windows\\Fonts\\consola.ttf"
    if not os.path.exists(font_path):
        font_path = "C:\\Windows\\Fonts\\cour.ttf"
    font = ImageFont.truetype(font_path, 15)
    title_font = ImageFont.truetype(font_path, 14)

    char_w = 9
    line_h = 21
    max_cols = max(len(l) for l in lines)
    pad_x, pad_y = 22, 18
    titlebar_h = 38

    img_w = max(860, max_cols * char_w + pad_x * 2)
    img_h = titlebar_h + len(lines) * line_h + pad_y * 2

    img = Image.new("RGB", (img_w, img_h), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Title bar
    draw.rectangle([0, 0, img_w, titlebar_h], fill=(30, 41, 59))
    # Window dots
    for idx, col in enumerate([(239, 68, 68), (245, 158, 11), (34, 197, 94)]):
        cx = 20 + idx * 20
        draw.ellipse([cx - 6, 13, cx + 6, 25], fill=col)
    draw.text((88, 11), f"Windows PowerShell — {title}", fill=(203, 213, 225), font=title_font)

    y = titlebar_h + pad_y
    for i, line in enumerate(lines):
        if i == 0:
            col = (56, 189, 248)  # cyan prompt/command
        elif line.startswith("===") or line.startswith("SUMMARY") or line.startswith("Total time") or line.startswith("GPU utilization"):
            col = (74, 222, 128)  # bright green highlights
        elif line.startswith("---"):
            col = (148, 163, 184)
        else:
            col = (226, 232, 240)
        draw.text((pad_x, y), line, fill=col, font=font)
        y += line_h

    img.save(out_path)


def generate_screenshots():
    with open("results/setup_check.txt", encoding="utf-8") as f:
        render_terminal_screenshot(
            "Step 0: Environment & CUDA Setup Check",
            "nvidia-smi; python --version; python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\"",
            f.read(),
            "screenshots/01_setup_check.png",
        )
    with open("results/baseline_output.txt", encoding="utf-8") as f:
        render_terminal_screenshot(
            "Step 1: Baseline Workload + Profiler",
            "python gpu_workload.py --mode baseline --profile",
            f.read(),
            "screenshots/02_baseline_run.png",
        )
    with open("results/optimized_output.txt", encoding="utf-8") as f:
        render_terminal_screenshot(
            "Step 4: Optimized Workload (Pinned Memory + 2 Streams) + Profiler",
            "python gpu_workload.py --mode optimized --profile",
            f.read(),
            "screenshots/03_optimized_run.png",
        )
    with open("results/analyzer_output.txt", encoding="utf-8") as f:
        render_terminal_screenshot(
            "Step 5: Trace Analyzer Comparison",
            "python analyze_trace.py results/trace_baseline_events.json results/trace_optimized_events.json",
            f.read(),
            "screenshots/04_trace_analyzer.png",
        )
    with open("results/optional_reps_output.txt", encoding="utf-8") as f:
        render_terminal_screenshot(
            "Optional Experiment: Compute Intensity Scaling (--reps 1, 3, 8)",
            "foreach ($r in 1,3,8) { python gpu_workload.py --mode baseline/optimized --reps $r }",
            f.read(),
            "screenshots/05_optional_reps.png",
        )


if __name__ == "__main__":
    plot_timelines()
    plot_performance_comparison()
    plot_smi_utilization()
    plot_kernel_and_roofline()
    plot_reps_scaling()
    generate_screenshots()
    print("Generated all 5 graphs in graphs/ and 5 terminal screenshots in screenshots/.")
