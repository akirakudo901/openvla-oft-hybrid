"""
inspect_benchmarks.py

Summarizes every installed MolmoSpaces benchmark: number of episodes and houses,
task classes, scene datasets / splits, robot, cameras, horizons, and example
instructions. Also dumps one full episode spec of FrankaPickHardBench so we can
see exactly what the policy will be given at evaluation time.

Usage:
    python inspect_benchmarks.py $MLSPACES_ASSETS_DIR/benchmarks
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

DUMP_FULL_SPEC_FOR = "FrankaPickHardBench"


def load_episodes(bench_dir: Path) -> list[dict]:
    """A benchmark is either one benchmark.json (list of specs) or house_*/episode_*.json files."""
    bench_json = bench_dir / "benchmark.json"
    if bench_json.is_file():
        data = json.loads(bench_json.read_text())
        return data if isinstance(data, list) else data.get("episodes", [data])
    return [json.loads(p.read_text()) for p in sorted(bench_dir.glob("house_*/episode_*.json"))]


def find_values(obj, key, out=None):
    """Collects every value stored under `key` anywhere in a nested dict/list."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            find_values(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            find_values(v, key, out)
    return out


def first(obj, key, default=None):
    values = find_values(obj, key)
    return values[0] if values else default


def camera_names(episode: dict) -> list[str]:
    cameras = first(episode, "cameras")
    if isinstance(cameras, dict):
        return sorted(cameras)
    if isinstance(cameras, list):
        return [c.get("name", "?") if isinstance(c, dict) else str(c) for c in cameras]
    return []


def summarize(bench_dir: Path, episodes: list[dict]):
    count = lambda key: dict(Counter(str(first(e, key)) for e in episodes))
    houses = {str(first(e, "house_index", first(e, "house_idx"))) for e in episodes}
    print(f"\n=== {bench_dir} ===")
    print(f"episodes: {len(episodes)}   houses: {len(houses)}")
    print(f"task_cls:      {count('task_cls')}")
    print(f"scene_dataset: {count('scene_dataset')}")
    print(f"data_split:    {count('data_split')}")
    print(f"robot_name:    {count('robot_name')}")
    print(f"cameras:       {camera_names(episodes[0])}")
    for key in ("task_horizon", "horizon", "policy_dt_ms", "action_spec", "action_type"):
        value = first(episodes[0], key)
        if value is not None:
            print(f"{key}: {value}")
    descriptions = [first(e, "task_description") for e in episodes[:3]]
    print(f"instructions:  {descriptions}")


def main():
    root = Path(sys.argv[1]).expanduser()
    # The installed benchmark dirs are symlinks into the cache; rglob does not follow them.
    bench_dirs = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        path = Path(dirpath)
        if "benchmark.json" in filenames:
            bench_dirs.add(path)
        elif path.name.startswith("house_") and any(f.startswith("episode_") for f in filenames):
            bench_dirs.add(path.parent)
    bench_dirs = sorted(bench_dirs)
    print(f"Found {len(bench_dirs)} benchmarks under {root}")
    dumped = False
    for bench_dir in bench_dirs:
        episodes = load_episodes(bench_dir)
        if not episodes:
            print(f"\n=== {bench_dir} === (no episodes found)")
            continue
        summarize(bench_dir, episodes)
        if not dumped and DUMP_FULL_SPEC_FOR in str(bench_dir) and "procthor-objaverse" in str(bench_dir):
            print(f"\n--- full episode spec [0] of {bench_dir.name} ---")
            print(json.dumps(episodes[0], indent=1, default=str)[:8000])
            dumped = True


if __name__ == "__main__":
    main()
