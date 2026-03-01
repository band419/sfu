from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from simulation.golden.model import golden_fp16

# Ensure built-in adapters are registered.
from simulation.dut import get_dut, list_duts
import simulation.dut.adapters  # noqa: F401


@dataclass(frozen=True)
class CompareRow:
    mode: str
    x_bits: int
    golden_bits: int
    dut_bits: int

    @property
    def match(self) -> bool:
        return self.golden_bits == self.dut_bits


def _parse_hex16(s: str) -> int:
    value = int(s, 16)
    if value < 0 or value > 0xFFFF:
        raise ValueError(f"hex16 out of range: {s}")
    return value


def load_vector_rows(csv_path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row["mode"].strip()
            x_bits = _parse_hex16(row["x_bits"])
            y_bits = _parse_hex16(row["y_bits"])
            rows.append((mode, x_bits, y_bits))
    return rows


def compare_with_dut(dut_name: str, rows: list[tuple[str, int, int]]) -> list[CompareRow]:
    dut = get_dut(dut_name)
    out: list[CompareRow] = []
    for mode, x_bits, vec_golden_bits in rows:
        # Recompute golden to avoid stale vector ambiguity.
        golden_bits = golden_fp16(mode, x_bits)
        if golden_bits != vec_golden_bits:
            raise RuntimeError(
                f"Vector golden mismatch for mode={mode}, x_bits=0x{x_bits:04X}: "
                f"csv=0x{vec_golden_bits:04X}, recomputed=0x{golden_bits:04X}"
            )
        dut_bits = dut.eval(mode, x_bits).y_bits
        out.append(CompareRow(mode=mode, x_bits=x_bits, golden_bits=golden_bits, dut_bits=dut_bits))
    return out


def summarize(results: list[CompareRow]) -> tuple[int, int]:
    total = len(results)
    mismatches = sum(1 for r in results if not r.match)
    return total, mismatches


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DUT model outputs against FP16 sin/cos golden.")
    parser.add_argument("--dut", required=True, help="Registered DUT adapter name")
    parser.add_argument(
        "--vectors",
        default=str(Path("simulation") / "vectors" / "directed_fp16_sincos.csv"),
        help="Path to vector CSV",
    )
    parser.add_argument("--max-report", type=int, default=10, help="Max mismatch rows to print")
    parser.add_argument("--list-duts", action="store_true", help="List available DUT adapters and exit")
    args = parser.parse_args()

    if args.list_duts:
        print("Available DUT adapters:")
        for name in list_duts():
            print(f"  - {name}")
        return

    rows = load_vector_rows(Path(args.vectors))
    results = compare_with_dut(args.dut, rows)
    total, mismatches = summarize(results)

    print(f"DUT: {args.dut}")
    print(f"Total rows: {total}")
    print(f"Mismatches: {mismatches}")

    if mismatches:
        print("Sample mismatches:")
        shown = 0
        for r in results:
            if r.match:
                continue
            print(
                f"  mode={r.mode}, x=0x{r.x_bits:04X}, golden=0x{r.golden_bits:04X}, dut=0x{r.dut_bits:04X}"
            )
            shown += 1
            if shown >= args.max_report:
                break
        raise SystemExit(1)


if __name__ == "__main__":
    main()
