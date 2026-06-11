"""Evaluate pi0.5 on LIBERO-PRO perturbed task suites and save episodes + policy embeddings.

Thin wrapper around eval_save_with_embeddings.py. LIBERO-PRO
(https://github.com/Zxy-MLlab/LIBERO-PRO) registers its perturbed task suites —
libero_{spatial,object,goal,10}_{object,swap,lan,task,env} plus the combined
``_temp`` suites — in the standard ``libero.libero.benchmark`` registry, so the
base eval loop works unchanged. This wrapper only:

* resolves the per-episode step budget for perturbed suite names (default: the
  base suite's budget, e.g. libero_spatial_env -> 220), and
* validates that the suite is registered and its bddl/init files exist.

Setup (LIBERO-PRO ships the same ``libero`` package name and replaces the
bundled LIBERO inside this uv env):

    uv pip install -e /path/to/LIBERO-PRO

then download the perturbed bddl/init files from
https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro into
LIBERO-PRO/libero/libero/{bddl_files,init_files}/ (or generate them with
LIBERO-PRO's perturbation.create_env).

Usage
-----
cd third_party/openpi
uv run examples/libero/eval_save_with_embeddings_pro.py \\
    --task-suite-name libero_spatial_env \\
    --output-dir /path/to/output \\
    --num-trials-per-task 20
"""

from __future__ import annotations

import dataclasses
import importlib.util
import logging
import pathlib
import sys

import tyro

# Load the base eval script as a module (examples/ is not a package).
_BASE_PATH = pathlib.Path(__file__).resolve().parent / "eval_save_with_embeddings.py"
_spec = importlib.util.spec_from_file_location("eval_save_with_embeddings", _BASE_PATH)
assert _spec is not None and _spec.loader is not None
_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _base
_spec.loader.exec_module(_base)

_BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")


@dataclasses.dataclass
class Args(_base.Args):
    task_suite_name: str = "libero_spatial_env"
    """LIBERO-PRO suite: <base>_<perturbation> with base in
    {libero_spatial, libero_object, libero_goal, libero_10} and perturbation in
    {object, swap, lan, task, env, temp} (temp = combined perturbations)."""

    max_steps: int | None = None
    """Per-episode env step budget. Default: the base suite's budget."""


def _resolve_max_steps(suite_name: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    for base_suite in _BASE_SUITES:
        if suite_name == base_suite or suite_name.startswith(base_suite + "_"):
            return _base._MAX_STEPS[base_suite]
    raise ValueError(f"Cannot infer a step budget for suite '{suite_name}'; pass --max-steps.")


def main(args: Args) -> None:
    from libero.libero import benchmark
    from libero.libero import get_libero_path

    args.task_suite_name = args.task_suite_name.lower()
    suite_cls = benchmark.get_benchmark_dict().get(args.task_suite_name)
    if suite_cls is None:
        raise ValueError(
            f"Task suite '{args.task_suite_name}' is not registered. Install LIBERO-PRO "
            "into this uv env first: uv pip install -e /path/to/LIBERO-PRO"
        )
    suite = suite_cls()
    if suite.n_tasks == 0:
        raise RuntimeError(f"Suite '{args.task_suite_name}' has no tasks registered.")

    # The task map is static; verify the suite's files actually exist on disk.
    task0 = suite.get_task(0)
    bddl0 = pathlib.Path(suite.get_task_bddl_file_path(0))
    init0 = pathlib.Path(get_libero_path("init_states")) / task0.problem_folder / task0.init_states_file
    missing = [p for p in (bddl0, init0) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing LIBERO-PRO task files for suite '{args.task_suite_name}': "
            + ", ".join(str(p) for p in missing)
            + "\nDownload bddl_files/init_files from "
            "https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro (or generate them with "
            "LIBERO-PRO's perturbation.create_env) and place them under "
            "LIBERO-PRO/libero/libero/{bddl_files,init_files}/."
        )

    _base._MAX_STEPS[args.task_suite_name] = _resolve_max_steps(args.task_suite_name, args.max_steps)
    _base.run_eval(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
