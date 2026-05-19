"""Evaluate pi0.5 on LIBERO and save episodes + policy embeddings for behavior graph analysis.

Runs the pi05_libero policy directly (no WebSocket server) and captures last-layer action embeddings
at each policy call.  Embeddings are replicated across the action chunk so the output is one vector
per environment timestep, matching the format expected by policy_doctor's clustering pipeline.

Output layout
-------------
<output_dir>/
  episodes/
    metadata.yaml              # episode_lengths, episode_successes
    ep0000_succ.pkl            # per-episode list of {t, action, success}
    ep0001_fail.pkl
    ...
  media/
    ep0000_succ.mp4            # video, compatible with behavior-graph tab playback
    ...
  policy_embeddings/
    pi05.npz                   # rollout_embeddings: (N_total_env_steps, hidden_dim) float32

Usage
-----
cd third_party/openpi
uv run examples/libero/eval_save_with_embeddings.py \\
    --task-suite-name libero_spatial \\
    --output-dir /path/to/output \\
    --num-trials-per-task 20
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pathlib
import pickle

import imageio
import jax
import jax.numpy as jnp
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
import tqdm
import tyro
import yaml

from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config
from openpi.shared import download

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_spatial"
    """Task suite: libero_spatial | libero_object | libero_goal | libero_10 | libero_90"""

    num_trials_per_task: int = 20
    """Rollouts per task in the suite."""

    output_dir: str = "data/pi05_libero"
    """Root output directory (will be created)."""

    resize_size: int = 224
    """Image size fed to the model."""

    replan_steps: int = 5
    """Execute this many env steps per action chunk (must be <= action_horizon=10)."""

    seed: int = 7

    checkpoint: str = "gs://openpi-assets/checkpoints/pi05_libero"
    """GCS path or local dir for the pi05_libero checkpoint."""

    config_name: str = "pi05_libero"
    """openpi TrainConfig name (must match the checkpoint)."""


_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_NUM_STEPS_WAIT = 10


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _save_results(
    episodes_dir: pathlib.Path,
    emb_dir: pathlib.Path,
    ep_lengths: list[int],
    ep_successes: list[bool],
    embeddings: list[np.ndarray],
) -> None:
    meta = {"episode_lengths": ep_lengths, "episode_successes": ep_successes}
    with open(episodes_dir / "metadata.yaml", "w") as f:
        yaml.safe_dump(meta, f)
    rollout_emb = np.stack(embeddings, axis=0).astype(np.float32)
    np.savez(emb_dir / "pi05.npz", rollout_embeddings=rollout_emb)


def run_eval(args: Args) -> None:
    np.random.seed(args.seed)
    rng = jax.random.key(args.seed)

    out = pathlib.Path(args.output_dir)
    episodes_dir = out / "episodes"
    media_dir = out / "media"
    emb_dir = out / "policy_embeddings"
    for d in (episodes_dir, media_dir, emb_dir):
        d.mkdir(parents=True, exist_ok=True)

    max_steps = _MAX_STEPS.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Load model
    logging.info("Loading pi05 model from %s ...", args.checkpoint)
    train_config = _config.get_config(args.config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    policy = _policy_config.create_trained_policy(train_config, checkpoint_dir)

    # JIT-compile embedding-capturing method
    model = policy._model
    jit_sample_emb = nnx_utils.module_jit(model.sample_actions_with_embeddings)

    # Task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info("Suite %s: %d tasks, %d trials each", args.task_suite_name, num_tasks, args.num_trials_per_task)

    all_ep_lengths: list[int] = []
    all_ep_successes: list[bool] = []
    all_embeddings: list[np.ndarray] = []  # per-env-timestep, accumulated
    global_ep_idx = 0

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        for trial_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            env.reset()
            obs = env.set_init_state(initial_states[trial_idx % len(initial_states)])

            t = 0
            action_plan: collections.deque = collections.deque()
            current_chunk_emb: np.ndarray | None = None  # embedding for the active chunk
            replay_images: list = []
            ep_actions: list = []
            ep_embeddings: list[np.ndarray] = []
            done = False

            while t < max_steps + _NUM_STEPS_WAIT:
                if t < _NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                )
                replay_images.append(img)

                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate((
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )),
                        "prompt": str(task_description),
                    }
                    # Apply input transforms (normalize, model transforms, etc.)
                    inputs = policy._input_transform(element)
                    inputs_batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
                    observation_jax = _model.Observation.from_dict(inputs_batched)

                    rng, sample_rng = jax.random.split(rng)
                    actions_jax, emb_jax = jit_sample_emb(sample_rng, observation_jax)

                    # Unnormalize actions via output transforms
                    raw_outputs = {"state": inputs_batched["state"], "actions": actions_jax}
                    raw_outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), raw_outputs)
                    outputs = policy._output_transform(raw_outputs)
                    action_chunk = outputs["actions"]  # [action_horizon, action_dim]

                    current_chunk_emb = np.asarray(emb_jax[0], dtype=np.float32)  # [hidden_dim]

                    assert len(action_chunk) >= args.replan_steps
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                ep_actions.append(action)
                ep_embeddings.append(current_chunk_emb)

                obs, reward, done, info = env.step(action.tolist())
                if done:
                    break
                t += 1

            ep_len = len(ep_actions)
            success = bool(done)
            suffix = "succ" if success else "fail"

            # Save episode pkl (minimal: action + success per step)
            ep_data = [{"t": i, "action": ep_actions[i].tolist(), "success": success} for i in range(ep_len)]
            ep_path = episodes_dir / f"ep{global_ep_idx:04d}_{suffix}.pkl"
            with open(ep_path, "wb") as f:
                pickle.dump(ep_data, f)

            # Save video
            if replay_images:
                video_path = media_dir / f"ep{global_ep_idx:04d}_{suffix}.mp4"
                imageio.mimwrite(str(video_path), [np.asarray(x) for x in replay_images], fps=10)

            # Accumulate embeddings for this episode
            all_embeddings.extend(ep_embeddings)
            all_ep_lengths.append(ep_len)
            all_ep_successes.append(success)
            global_ep_idx += 1

            logging.info(
                "ep%04d %s (len=%d) — total: %d/%d",
                global_ep_idx - 1, suffix, ep_len,
                sum(all_ep_successes), len(all_ep_successes),
            )

        env.close()

        # Save intermediate checkpoint after each task so clustering can start early.
        _save_results(episodes_dir, emb_dir, all_ep_lengths, all_ep_successes, all_embeddings)
        logging.info("Checkpoint saved after task %d (%d episodes total)", task_id, global_ep_idx)

    # Final save (same as last checkpoint).
    _save_results(episodes_dir, emb_dir, all_ep_lengths, all_ep_successes, all_embeddings)

    total = len(all_ep_successes)
    n_succ = sum(all_ep_successes)
    rollout_emb = np.stack(all_embeddings, axis=0)
    logging.info(
        "Done. %d episodes, success rate %.1f%% (%d/%d)",
        total, 100.0 * n_succ / total if total > 0 else 0.0, n_succ, total,
    )
    logging.info("Embeddings: %s -> %s", rollout_emb.shape, emb_dir / "pi05.npz")
    logging.info("Episodes:   %s", episodes_dir)
    logging.info("Videos:     %s", media_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    run_eval(args)
