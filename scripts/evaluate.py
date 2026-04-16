"""Backward-compatible wrapper for evaluation CLI."""

from evaluation.cli.evaluate import build_model, build_sampler_fn, main, parse_args

__all__ = ["build_model", "build_sampler_fn", "parse_args", "main"]


if __name__ == "__main__":
    main()
