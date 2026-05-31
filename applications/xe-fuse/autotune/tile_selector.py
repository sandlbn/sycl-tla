"""
xe-fuse autotune: tile shape selector.

Selects the best tile shape for a GEMM based on (M, N, K) dimensions.
Rules derived from empirical sweep of 7 tiles × 32 shapes × 6 M values
on Intel Arc Pro B70 (BMG G31), 444 benchmark points.

Usage:
    from tile_selector import select_tile

    tile = select_tile(M=128, N=4096, K=4096)
    # Returns "_128, _128, _32"

    tile = select_tile(M=32, N=896, K=896)
    # Returns "_64, _64, _32"
"""

import csv
import os
from pathlib import Path

# Tile shape options (tile_M x tile_N x tile_K)
TILES = {
    "256x256x32": "_256, _256, _32",
    "256x128x32": "_256, _128, _32",
    "128x256x32": "_128, _256, _32",
    "128x128x32": "_128, _128, _32",
    "64x256x32":  "_64, _256, _32",
    "64x128x32":  "_64, _128, _32",
    "64x64x32":   "_64, _64, _32",
}

# Heuristic tile selection rules from sweep analysis.
# K is always 32 (XMX constraint for bf16 on BMG).
def select_tile(M: int, N: int, K: int, kernel: str = "bare") -> str:
    """Select optimal tile shape string for CUTLASS cute::Shape<>.

    Args:
        M: number of rows (sequence length / batch)
        N: output columns (hidden dim, FFN dim, etc.)
        K: reduction dim (input hidden dim)
        kernel: "bare", "k1", "k2", or "k4" (affects register pressure)

    Returns:
        CUTLASS tile shape string, e.g. "_128, _128, _32"
    """
    # Tile M: match the problem M, capped at 256 and N
    if M <= 64:
        tile_m = 64
    elif M <= 128:
        tile_m = 128 if N >= 1024 else 64
    elif M <= 256:
        tile_m = 256 if N >= 2048 else 128
    else:
        tile_m = 256

    # Never pick tile_m > M or tile_m > N
    tile_m = min(tile_m, max(64, M), max(64, N))

    # Tile N: depends on N dimension and tile_m
    if N <= 256:
        tile_n = 64
    elif N <= 512:
        tile_n = 128 if tile_m <= 128 else 64
    elif N <= 1024:
        if tile_m <= 64:
            tile_n = 128
        elif tile_m <= 128:
            tile_n = 128
        else:
            tile_n = 128
    elif N <= 4096:
        if tile_m <= 64:
            tile_n = 128 if N <= 2048 else 256
        elif tile_m <= 128:
            tile_n = 128 if N <= 2048 else 256
        else:
            tile_n = 128
    else:
        # Large N (FFN): prefer wide tiles
        if tile_m <= 64:
            tile_n = 256
        elif tile_m <= 128:
            tile_n = 256
        else:
            tile_n = 256 if M >= 512 else 128

    # K2 (SwiGLU/GeGLU) has more register pressure — prefer smaller tile_n
    if kernel in ("k2", "k2_geglu") and tile_m * tile_n > 256 * 128:
        if tile_n > 128 and tile_m >= 256:
            tile_n = 128

    # K4 (RMSNorm+RoPE) has the highest register pressure.
    # Validated tiles from sweep: 64x128, 64x64, 128x128, 128x256, 256x128, 256x256.
    # 64x256 does NOT compile for K4.
    if kernel in ("k4", "k4v2"):
        if tile_m <= 64:
            tile_n = min(tile_n, 128)
        elif tile_m <= 128:
            tile_n = min(tile_n, 256)
        # 256x128 and 256x256 are both fine

    # Never pick tile_n > N
    tile_n = min(tile_n, max(64, N))

    key = f"{tile_m}x{tile_n}x32"
    return TILES.get(key, "_256, _256, _32")


def select_tile_from_csv(M: int, N: int, K: int, kernel: str = "Bare_GEMM(bf16)") -> str:
    """Look up the best tile from the sweep CSV if available, else fall back to heuristic."""
    csv_path = Path(__file__).parent.parent / "tests" / "best_tiles.csv"
    if not csv_path.exists():
        return select_tile(M, N, K, kernel)

    best_tf = 0.0
    best_tile = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row["kernel"] == kernel and
                int(row["M"]) == M and
                int(row["N"]) == N and
                int(row["K"]) == K):
                tf = float(row["tflops"])
                if tf > best_tf:
                    best_tf = tf
                    best_tile = row["best_tile"]

    if best_tile and best_tile in TILES:
        return TILES[best_tile]

    return select_tile(M, N, K, kernel)


def tile_shape_str(M: int, N: int, K: int, kernel: str = "bare") -> str:
    """Convenience: returns the tile shape for generate_kernel/pipeline use."""
    return select_tile(M, N, K, kernel)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="xe-fuse tile selector")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--kernel", default="bare")
    args = parser.parse_args()

    tile = select_tile(args.m, args.n, args.k, args.kernel)
    print(f"M={args.m} N={args.n} K={args.k} kernel={args.kernel} -> tile={tile}")
