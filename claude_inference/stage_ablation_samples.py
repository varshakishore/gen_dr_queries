#!/usr/bin/env python3
"""
Stage ablation-arm sample files for joint clustering with strategy_feedback_module.py.

Why this exists: the feedback module's --runs discovery globs at most
`*/*/sample_*.json` (three levels), but research_loop.py writes
`<arm>/round_NN/<system>/<prompt>/sample_NNN.json` (four). Pointing --runs at an arm
root therefore finds nothing, and the module reports it as
"no usable examples: every entry lacked a non-empty `strategy`".

Pointing --runs at each round dir does find the files, but the module's source_run
label is "<system>/<prompt>" whenever the grandparent is a round dir, so every arm
collapses into the same buckets and the arms become indistinguishable in the output.

This script symlinks every arm's sample files into one flat directory per arm:

    <out>/<arm>/sample_<round>_<prompt>_<nnn>.json  ->  original

so the direct glob matches and source_run is the arm name. Originals are never
touched; only symlinks are created. An existing output dir is refused.

  python stage_ablation_samples.py --out runs/abl_staging \
      runs/abl_full runs/abl_exploit_fixed runs/abl_explore_fixed runs/abl_free
"""
import argparse
import sys
from pathlib import Path


def stage(arm_dir: Path, out_dir: Path) -> int:
    """Symlink every sample_*.json under `arm_dir` into `out_dir`, flattened."""
    dest = out_dir / arm_dir.name
    dest.mkdir(parents=True)
    n = 0
    for src in sorted(arm_dir.glob("round_*/*/*/sample_*.json")):
        prompt = src.parent.name                      # exploit | explore
        rnd = src.parent.parent.parent.name           # round_NN
        link = dest / f"sample_{rnd}_{prompt}_{src.stem.split('_')[-1]}.json"
        if link.exists():
            sys.exit(f"name collision: {link} (two sources map to one staged name)")
        link.symlink_to(src.resolve())
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+", type=Path, help="arm run dirs")
    ap.add_argument("--out", type=Path, required=True, help="staging dir (must not exist)")
    args = ap.parse_args()

    if args.out.exists():
        sys.exit(f"{args.out} exists; refusing to overwrite (delete it or pick another --out)")
    for arm in args.arms:
        if not arm.is_dir():
            sys.exit(f"not a directory: {arm}")

    total = 0
    for arm in args.arms:
        n = stage(arm, args.out)
        total += n
        print(f"  {arm.name:24} {n:4} sample files -> {args.out / arm.name}")
    print(f"\n{total} files staged under {args.out}")
    print("\nNow run:\n"
          f"  python strategy_feedback_module.py \\\n"
          f"    --runs {' '.join(str(args.out / a.name) for a in args.arms)} \\\n"
          "    --seeds-file runs/abl_full/round_00/example_strategy_pool.txt \\\n"
          "    --deciding-only --keep-run-duplicates \\\n"
          "    --cluster-provider openai --cluster-model gpt-5.6-luna \\\n"
          "    --out runs/abl_joint_clusters.json")


if __name__ == "__main__":
    main()
