"""GENESIS paper-preparation evaluation pipeline.

Submodules:
    normalization  – affine ↔ log10 ↔ physical converters (matches training).
    data_loaders   – LH (pre-normalized test split) + CV/1P/EX (raw + normalize).
    generator      – checkpoint + flow-matching sampler wrapper.
    metrics        – P(k), cross P, r(k), PDF — raw numbers, no pass/fail.
    cv_floor       – σ_CV(k) computation/caching from CV set.
    evaluator      – per-condition long-format rows + per-cond NPZ writer.
    plotting       – per-condition / distribution plots.
"""
