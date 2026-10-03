"""
molmospaces_utils.py

Shared helpers for MolmoBot-Data / MolmoSpaces: reading raw episodes for RLDS
conversion, and the observation/action conventions that training and benchmark
evaluation must agree on. Deliberately free of TensorFlow and torch imports so it
can be used in multiprocessing workers and in the MolmoSpaces eval env.

Conventions (mirroring the MolmoSpaces reference Pi policy, `pi_policy.py`):
    image        224x224 letterboxed (aspect-preserving resize + black padding)
    state  (8,)  7 arm joint angles [rad] + gripper opening in [0, 1]
    action (8,)  7 absolute target arm joint angles [rad] + gripper command in {0, 1} (1 = close)
"""

import json
import zlib
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from PIL import Image

IMAGE_SIZE = 224
WRIST_CAMERA = "wrist_camera_zed_mini"
EXO_CAMERAS = (
    "droid_shoulder_light_randomization",
    "randomized_zed2_analogue_1",
    "randomized_zed2_analogue_2",
    "randomized_gopro_analogue_1",
)
GRIPPER_QPOS_MAX = 0.824033  # fully closed finger joint position (qpos), same as pi_policy.py
GRIPPER_CMD_MAX = 255.0  # commanded gripper value for "close"


def decode_json_datum(datum: np.ndarray):
    """Decodes a null-padded JSON bytes array (how MolmoBot-Data stores actions and qpos)."""
    return json.loads(datum.tobytes().decode("utf-8").rstrip("\x00"))


def resize_with_pad(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Aspect-preserving bilinear resize into a size x size black canvas (as openpi's resize_with_pad)."""
    h, w = image.shape[:2]
    ratio = max(w / size, h / size)
    new_w, new_h = int(w / ratio), int(h / ratio)
    resized = Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return np.asarray(canvas)


def qpos_to_state(qpos: dict) -> np.ndarray:
    gripper = np.clip(qpos["gripper"][0] / GRIPPER_QPOS_MAX, 0.0, 1.0)
    return np.asarray(list(qpos["arm"][:7]) + [gripper], dtype=np.float32)


def command_to_action(command: dict) -> np.ndarray:
    gripper = command["gripper"][0] / GRIPPER_CMD_MAX
    return np.asarray(list(command["arm"][:7]) + [gripper], dtype=np.float32)


def find_episodes(split_dir: Path) -> list[tuple[str, str]]:
    """Lists (h5 path, traj key) for every valid, successful trajectory under a raw split dir."""
    episodes = []
    for h5_path in sorted(Path(split_dir).glob("house_*/trajectories_*.h5")):
        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted((k for k in f.keys() if k.startswith("traj_")), key=lambda k: int(k.split("_")[1]))
            valid = f["valid_traj_mask"][()] if "valid_traj_mask" in f else [True] * len(traj_keys)
            for key in traj_keys:
                if valid[int(key.split("_")[1])] and bool(f[key]["success"][-1]):
                    episodes.append((str(h5_path), key))
    return episodes


def pick_exo_camera(h5_path: str, traj_key: str, seed: int) -> str:
    """Deterministic per-episode random choice of the third-person camera."""
    rng = np.random.default_rng(zlib.crc32(f"{seed}/{h5_path}/{traj_key}".encode()))
    return EXO_CAMERAS[rng.integers(len(EXO_CAMERAS))]


def load_episode(h5_path: str, traj_key: str, exo_camera: str) -> dict | None:
    """
    Reads one trajectory as aligned numpy arrays, or returns None (with a reason) if it is unusable.

    Alignment: MolmoBot stores T observations and T commanded actions where action[0] is a dummy and
    action[t + 1] is what was commanded after observing step t. We emit T - 1 steps
    (observation t, action t + 1) and drop the final observation.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        traj = f[traj_key]
        commands = [decode_json_datum(d) for d in traj["actions"]["commanded_action"][()]]
        qpos = [decode_json_datum(d) for d in traj["obs"]["agent"]["qpos"][()]]
        video_names = {
            cam: bytes(traj["obs"]["sensor_data"][cam][()]).rstrip(b"\x00").decode()
            for cam in (exo_camera, WRIST_CAMERA)
        }
        instruction = json.loads(traj["obs_scene"][()].decode())["task_description"]

    num_obs = len(qpos)
    if num_obs < 3:
        return {"skip": f"too short ({num_obs} steps)"}

    frames = {}
    for cam, name in video_names.items():
        video = iio.imread(h5_path.parent / name, plugin="pyav")
        if len(video) != num_obs:
            return {"skip": f"{cam} has {len(video)} frames, expected {num_obs}"}
        frames[cam] = np.stack([resize_with_pad(frame) for frame in video[: num_obs - 1]])

    states = np.stack([qpos_to_state(q) for q in qpos])
    actions = np.stack([command_to_action(c) for c in commands[1:]])
    return {
        "image": frames[exo_camera],
        "wrist_image": frames[WRIST_CAMERA],
        "state": states[:-1],
        "action": actions,
        "next_state": states[1:],  # only used for the alignment sanity check
        "instruction": instruction,
        "exo_camera": exo_camera,
        "file_path": f"{h5_path}:{traj_key}",
    }


def load_episode_task(args) -> dict:
    """Picklable worker entry point: args = (h5_path, traj_key, seed)."""
    h5_path, traj_key, seed = args
    try:
        episode = load_episode(h5_path, traj_key, pick_exo_camera(h5_path, traj_key, seed))
    except Exception as e:  # corrupt file/video: report and skip instead of killing the whole conversion
        episode = {"skip": f"{type(e).__name__}: {e}"}
    episode["key"] = f"{Path(h5_path).parent.name}/{Path(h5_path).stem}/{traj_key}"
    return episode
