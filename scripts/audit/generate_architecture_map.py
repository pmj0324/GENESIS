from __future__ import annotations

import argparse
import ast
import json
from collections import defaultdict
from pathlib import Path


CORE_PREFIXES = {
    "train",
    "evaluate",
    "sample_cv",
    "analysis.",
    "dataloader.",
    "diffusion.",
    "flow_matching.",
    "models.",
    "training.",
    "utils.",
}


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    if rel.name == "__init__.py":
        return ".".join(rel.parts[:-1])
    return ".".join(rel.with_suffix("").parts)


def _is_core_module(mod: str) -> bool:
    return mod in {"train", "evaluate", "sample_cv"} or any(
        mod.startswith(prefix) for prefix in CORE_PREFIXES if prefix.endswith(".")
    )


def _internal_target(name: str, modules: set[str]) -> str | None:
    if not name:
        return None
    parts = name.split(".")
    for i in range(len(parts), 0, -1):
        cand = ".".join(parts[:i])
        if cand in modules or any(m.startswith(cand + ".") for m in modules):
            return cand
    return None


def _build_edges(root: Path) -> tuple[dict[str, set[str]], set[str]]:
    py_files = [
        p
        for p in root.rglob("*.py")
        if ".git/" not in str(p)
        and "/runs/" not in str(p)
        and "/outputs/" not in str(p)
        and "/__pycache__/" not in str(p)
    ]
    mod_by_file = {p: _module_name(root, p) for p in py_files}
    modules = set(mod_by_file.values())
    edges: dict[str, set[str]] = defaultdict(set)

    for path, mod in mod_by_file.items():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    tgt = _internal_target(alias.name, modules)
                    if tgt and tgt != mod:
                        edges[mod].add(tgt)
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.module is None:
                    base_parts = mod.split(".")[: -node.level]
                    base = ".".join(base_parts)
                elif node.level:
                    base_parts = mod.split(".")[: -node.level]
                    base_mod = node.module or ""
                    base = ".".join(base_parts + [base_mod] if base_mod else base_parts)
                else:
                    base = node.module or ""

                tgt = _internal_target(base, modules)
                if tgt and tgt != mod:
                    edges[mod].add(tgt)

                for alias in node.names:
                    if alias.name == "*":
                        continue
                    full = (base + "." + alias.name).strip(".")
                    tgt2 = _internal_target(full, modules)
                    if tgt2 and tgt2 != mod:
                        edges[mod].add(tgt2)

    return edges, modules


def _cycles(edges: dict[str, set[str]], modules: set[str]) -> list[list[str]]:
    visited: set[str] = set()
    in_stack: set[str] = set()
    stack: list[str] = []
    cycles: set[tuple[str, ...]] = set()

    def dfs(u: str) -> None:
        visited.add(u)
        in_stack.add(u)
        stack.append(u)
        for v in edges.get(u, set()):
            if v not in visited:
                dfs(v)
            elif v in in_stack:
                i = stack.index(v)
                cyc = tuple(stack[i:] + [v])
                cycles.add(cyc)
        stack.pop()
        in_stack.remove(u)

    for m in sorted(modules):
        if m not in visited:
            dfs(m)

    return [list(c) for c in sorted(cycles)]


def _group(mod: str) -> str:
    return mod.split(".")[0]


def _group_edges(edges: dict[str, set[str]], selected: set[str]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for src, tgts in edges.items():
        if src not in selected:
            continue
        gs = _group(src)
        for tgt in tgts:
            if tgt not in selected:
                continue
            gt = _group(tgt)
            if gs != gt:
                out[gs].add(gt)
    return {k: sorted(v) for k, v in sorted(out.items())}


def _serialize_edges(edges: dict[str, set[str]], selected: set[str]) -> dict[str, list[str]]:
    serial = {}
    for src in sorted(selected):
        tgts = sorted(t for t in edges.get(src, set()) if t in selected)
        if tgts:
            serial[src] = tgts
    return serial


def _to_markdown(
    core_edges: dict[str, list[str]],
    full_group_edges: dict[str, list[str]],
    core_cycles: list[list[str]],
    full_cycles: list[list[str]],
) -> str:
    lines: list[str] = []
    lines.append("# GENESIS Architecture Map")
    lines.append("")
    lines.append("## Core Import Graph")
    if not core_edges:
        lines.append("- none")
    else:
        for src, tgts in core_edges.items():
            lines.append(f"- `{src}` -> {', '.join(f'`{t}`' for t in tgts)}")
    lines.append("")
    lines.append("## Core Cycles")
    if core_cycles:
        for cyc in core_cycles:
            lines.append(f"- `{' -> '.join(cyc)}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Full Group Dependency Map")
    if not full_group_edges:
        lines.append("- none")
    else:
        for src, tgts in full_group_edges.items():
            lines.append(f"- `{src}` -> {', '.join(f'`{t}`' for t in tgts)}")
    lines.append("")
    lines.append("## Full Cycles")
    if full_cycles:
        for cyc in full_cycles:
            lines.append(f"- `{' -> '.join(cyc)}`")
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GENESIS architecture dependency map")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args()

    root = args.repo_root.resolve()
    edges, modules = _build_edges(root)
    core_modules = {m for m in modules if _is_core_module(m)}

    core_edge_map = _serialize_edges(edges, core_modules)
    core_cycles = _cycles(edges, core_modules)
    full_cycles = _cycles(edges, modules)
    full_group_edge_map = _group_edges(edges, modules)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    args.out_md.write_text(
        _to_markdown(core_edge_map, full_group_edge_map, core_cycles, full_cycles),
        encoding="utf-8",
    )
    args.out_json.write_text(
        json.dumps(
            {
                "core_edges": core_edge_map,
                "core_cycles": core_cycles,
                "full_group_edges": full_group_edge_map,
                "full_cycles": full_cycles,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    print(f"saved: {args.out_md}")
    print(f"saved: {args.out_json}")


if __name__ == "__main__":
    main()
