from __future__ import annotations

"""ULP compare configuration template (configuration only, no executable entry).

Edit this file to define:
- which single golden to use
- which DUT models to compare
- input-domain sampling parameters
"""

from simulation.ulp_compare_models import CompareConfig


# Single golden for all models.
# - "golden-fp16" means exact golden model in simulation.golden.model.golden_fp16
# - You may also use a registered DUT adapter name as reference if needed.
GOLDEN = "golden-fp16"


# Models under test (all compared against GOLDEN above).
_interp_segments = [16, 32, 64, 128]
_interp_prefixes = ["u", "nu"]
_interp_suffixes = ["lin", "quad"]

_interp_models = [
    f"qflow-ph-interp-{pref}{seg}-{suf}"
    for seg in _interp_segments
    for pref in _interp_prefixes
    for suf in _interp_suffixes
]

MODELS = [
    "golden-mirror",
    "qflow-fixedconst-fp16",
    "qflow-fixedconst-fp32",
    "qflow-fixedconst-fp128",
    "qflow-ph",
    *_interp_models,
]


# Input domain configuration.
# If EXHAUSTIVE=True, SAMPLES/SEED are ignored and all 65536 fp16 inputs are used.
EXHAUSTIVE = False
SAMPLES = 200000
SEED = 20260223


# Unified config object consumed by run_stats_from_config(...)
COMPARE_CONFIG = CompareConfig(
    golden=GOLDEN,
    models=MODELS,
    exhaustive=EXHAUSTIVE,
    samples=SAMPLES,
    seed=SEED,
)
