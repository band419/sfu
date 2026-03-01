from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable

from .fp16 import decode_fp16_bits
from simulation.golden.model import golden_fp16


def _hex16(v: int) -> str:
    return f"0x{v:04X}"


def directed_inputs() -> list[int]:
    specials = [
        0x0000,
        0x8000,
        0x7C00,
        0xFC00,
        0x7E00,
        0x7D00,
        0xFE00,
    ]

    boundaries = [
        0x0001,  # min subnormal
        0x03FF,  # max subnormal
        0x0400,  # min normal
        0x3C00,  # +1.0
        0xBC00,  # -1.0
        0x3555,  # ~1/3
        0x3E00,  # 1.5
        0x4200,  # 3.0
        0x7BFF,  # max finite
        0xFBFF,  # min finite (negative)
    ]

    return specials + boundaries


def random_inputs(count: int, seed: int = 20260222) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(0, 0x10000) for _ in range(count)]


def generate_vectors(samples: Iterable[int], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "x_bits", "y_bits", "class_tag"])
        for x_bits in samples:
            x_class = decode_fp16_bits(x_bits).cls.value
            for mode in ("sin", "cos"):
                y_bits = golden_fp16(mode, x_bits)
                writer.writerow([mode, _hex16(x_bits), _hex16(y_bits), x_class])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    out_csv = project_root / "simulation" / "vectors" / "directed_fp16_sincos.csv"

    inputs = directed_inputs() + random_inputs(count=64)
    generate_vectors(inputs, out_csv)
    print(f"Wrote vectors to: {out_csv}")


if __name__ == "__main__":
    main()
