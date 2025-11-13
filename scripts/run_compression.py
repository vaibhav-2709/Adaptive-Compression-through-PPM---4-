import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
from compression.ppm import PPMCompressor
from compression.adaptive_huffman import AdaptiveHuffman


# ---------- Setup ----------
def ensure_dirs():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out = os.path.join(base, "output")
    os.makedirs(out, exist_ok=True)
    return {
        "project_root": base,
        "data_path": os.path.join(base, "data", "smart_home_energy_consumption_large.csv"),
        "out_dir": out
    }


# ---------- Visualization ----------
def plot_comparisons(results, out_dir):
    """Generate multiple visualizations for compression results."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams.update({"font.size": 10, "figure.dpi": 110})

    algos = list(results.keys())
    ratios = np.array([r["compression_ratio"] for r in results.values()])
    comp_times = np.array([r["compression_time"] for r in results.values()])
    decomp_times = np.array([r["decompression_time"] for r in results.values()])
    space_saved = (1 - ratios) * 100
    throughput = 1 / (comp_times + 1e-9)  # pseudo relative speed

    # ===== Basic Comparative Graphs =====
    plots = [
        ("Compression Ratio (Compressed/Original)", ratios, "Ratio", "comparison_ratios.png"),
        ("Compression Time Comparison", comp_times, "Time (s)", "comparison_times.png"),
        ("Decompression Time Comparison", decomp_times, "Time (s)", "comparison_decompression_times.png"),
        ("Percentage Space Saved", space_saved, "% Saved", "comparison_space_saved.png"),
    ]
    for title, vals, ylabel, filename in plots:
        plt.figure(figsize=(7, 5))
        bars = plt.bar(algos, vals)
        plt.bar_label(bars, fmt="%.3f", padding=3)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()

    # ===== Individual Detailed Metrics =====
    for algo, r in results.items():
        plt.figure(figsize=(6, 4))
        metrics = {
            "Compression Ratio": r["compression_ratio"],
            "Compression Time (s)": r["compression_time"],
            "Decompression Time (s)": r["decompression_time"],
            "Space Saved (%)": (1 - r["compression_ratio"]) * 100,
        }
        plt.bar(metrics.keys(), metrics.values(),
                color=["#4e79a7", "#f28e2b", "#76b7b2", "#59a14f"])
        plt.title(f"{algo} - Detailed Metrics")
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{algo.lower().replace(' ', '_')}_detailed.png"))
        plt.close()

    # ===== Trade-off Scatter: Ratio vs Time =====
    plt.figure(figsize=(7, 5))
    plt.scatter(comp_times, ratios, s=150, color="royalblue")
    for i, algo in enumerate(algos):
        plt.text(comp_times[i] + 0.01, ratios[i], algo, fontsize=9)
    plt.xlabel("Compression Time (s)")
    plt.ylabel("Compression Ratio (lower is better)")
    plt.title("Compression Time vs Compression Ratio Trade-off")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compression_tradeoff.png"))
    plt.close()

    # ===== Hybrid Bar + Line =====
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.bar(algos, space_saved, color="skyblue", label="Space Saved (%)")
    ax2 = ax1.twinx()
    ax2.plot(algos, comp_times, color="red", marker="o", label="Compression Time (s)")
    ax1.set_ylabel("Space Saved (%)")
    ax2.set_ylabel("Time (s)")
    plt.title("Space Saved vs Compression Time (Hybrid View)")
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "hybrid_space_time.png"))
    plt.close()

    # ===== Normalized Metrics =====
    metrics = np.vstack([ratios, comp_times, decomp_times, space_saved])
    normalized = (metrics - metrics.min(axis=1, keepdims=True)) / (
        metrics.max(axis=1, keepdims=True) - metrics.min(axis=1, keepdims=True))
    labels = ["Ratio", "Comp Time", "Decomp Time", "Space Saved"]
    plt.figure(figsize=(8, 5))
    for i, label in enumerate(labels):
        plt.plot(algos, normalized[i], marker="o", label=label)
    plt.legend()
    plt.title("Normalized Metric Comparison (0–1 Scale)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "normalized_metrics.png"))
    plt.close()

    # ===== Radar Charts (Speed vs Efficiency) =====
    from math import pi
    categories = ["Efficiency (1/Ratio)", "Speed (1/Time)", "Space Saved"]
    num_vars = len(categories)
    for i, algo in enumerate(algos):
        values = [1 / ratios[i], throughput[i] / throughput.max(), space_saved[i] / 100]
        values += values[:1]
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        plt.figure(figsize=(5, 5))
        ax = plt.subplot(111, polar=True)
        plt.xticks(angles[:-1], categories, color="grey", size=8)
        ax.plot(angles, values, linewidth=2, linestyle="solid", label=algo)
        ax.fill(angles, values, alpha=0.25)
        plt.title(f"{algo} - Speed vs Efficiency Radar")
        plt.legend(loc="upper right", bbox_to_anchor=(0.9, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{algo.lower().replace(' ', '_')}_radar.png"))
        plt.close()

    # ===== Space Saved vs Ratio Scatter =====
    plt.figure(figsize=(7, 5))
    plt.scatter(space_saved, ratios, s=150, color="orange")
    for i, algo in enumerate(algos):
        plt.text(space_saved[i] + 0.2, ratios[i], algo, fontsize=9)
    plt.xlabel("Space Saved (%)")
    plt.ylabel("Compression Ratio")
    plt.title("Space Saved vs Compression Ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "space_saved_vs_ratio.png"))
    plt.close()

    # ===== Throughput Chart =====
    throughput_MBps = (1 / comp_times) * 1e6
    plt.figure(figsize=(7, 5))
    plt.bar(algos, throughput_MBps, color="lightgreen")
    plt.title("Relative Compression Throughput")
    plt.ylabel("Relative MB/s (scaled)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "compression_throughput.png"))
    plt.close()

    print(f" Saved {8 + len(results)} detailed visualizations in {out_dir}")


# ---------- Helpers ----------
def save_bits_file(data_bytes, path):
    bits = "".join(f"{b:08b}" for b in data_bytes)
    with open(path, "w", encoding="utf-8") as f:
        f.write(bits)
    print(f" Binary bitstring written to {path}")


def export_encoding_map(data_bytes, out_path):
    freq = Counter(data_bytes)
    total = sum(freq.values())
    enc_map = {
        "total_symbols": total,
        "unique_symbols": len(freq),
        "symbols": {
            str(sym): {
                "char": chr(sym) if 32 <= sym < 127 else f"\\x{sym:02x}",
                "count": cnt,
                "probability": round(cnt / total, 6)
            }
            for sym, cnt in freq.items()
        }
    }
    with open(out_path, "w") as f:
        json.dump(enc_map, f, indent=2)
    print(f" Encoding map written to {out_path}")


# ---------- Main ----------
def main():
    paths = ensure_dirs()
    data_path = paths["data_path"]
    out_dir = paths["out_dir"]

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    print(" Reading dataset...")
    with open(data_path, "rb") as f:
        data = f.read()
    orig_size = len(data)
    print(f" Read complete. Size: {orig_size:,} bytes")

    results = {}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ===================================================
    #  Adaptive Huffman (0.5 MB limit)
    # ===================================================
    print("\n Starting Adaptive Huffman Compression...")
    if orig_size > 512_000:
        print(" Dataset large → using only first 0.5 MB for Adaptive Huffman.")
        data_ah = data[:512_000]
    else:
        data_ah = data

    ah = AdaptiveHuffman()
    t0 = time.time()
    try:
        comp_ah = ah.compress_bytes(data_ah)
        t1 = time.time()
        decomp_ah = ah.decompress_bytes(comp_ah)
        t2 = time.time()
    except Exception as e:
        print(f" Adaptive Huffman failed: {e}")
        comp_ah, decomp_ah, t1, t2 = b"", b"", t0, t0

    if comp_ah:
        comp_ratio_ah = len(comp_ah) / len(data_ah)
        lossless_ah = data_ah == decomp_ah
        results["Adaptive Huffman"] = {
            "compression_ratio": round(comp_ratio_ah, 6),
            "compression_time": round(t1 - t0, 6),
            "decompression_time": round(t2 - t1, 6),
            "lossless": lossless_ah
        }

        ah_base = os.path.join(out_dir, f"adaptive_huffman_{ts}")
        with open(ah_base + ".bin", "wb") as f:
            f.write(comp_ah)
        save_bits_file(comp_ah, ah_base + "_bits.txt")
        with open(ah_base + "_decompressed.csv", "wb") as f:
            f.write(decomp_ah)
        export_encoding_map(data_ah, ah_base + "_encoding_map.json")
        print(f" Adaptive Huffman Done. Ratio={comp_ratio_ah:.4f}, Lossless={lossless_ah}")
    else:
        print(" Skipped Adaptive Huffman (too slow or failed).")

    # ===================================================
    #  PPM-C (Order 4)
    # ===================================================
    print("\n Starting PPM-C (Order 4) Compression...")
    ppm = PPMCompressor(order=4)
    t3 = time.time()
    comp_ppm = ppm.compress_bytes(data)
    t4 = time.time()
    decomp_ppm = ppm.decompress_bytes(comp_ppm)
    t5 = time.time()

    comp_ratio_ppm = len(comp_ppm) / orig_size
    lossless_ppm = data == decomp_ppm
    results["PPM-C (Order 4)"] = {
        "compression_ratio": round(comp_ratio_ppm, 6),
        "compression_time": round(t4 - t3, 6),
        "decompression_time": round(t5 - t4, 6),
        "lossless": lossless_ppm
    }

    ppm_base = os.path.join(out_dir, f"ppm_c_order4_{ts}")
    with open(ppm_base + ".bin", "wb") as f:
        f.write(comp_ppm)
    save_bits_file(comp_ppm, ppm_base + "_bits.txt")
    with open(ppm_base + "_decompressed.csv", "wb") as f:
        f.write(decomp_ppm)
    export_encoding_map(data, ppm_base + "_encoding_map.json")

    print(f" PPM-C Done. Ratio={comp_ratio_ppm:.4f}, Lossless={lossless_ppm}")

    # ===================================================
    #  Save & Visualize
    # ===================================================
    results_path = os.path.join(out_dir, f"compression_comparison_{ts}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    plot_comparisons(results, out_dir)

    print("\n Compression comparison complete!")
    print(f"Results saved in: {results_path}")


if __name__ == "__main__":
    main()
