"""
molmospaces_oft_policy.py

MolmoSpaces policy + eval config that query an OpenVLA-OFT policy server (oft_policy_server.py).
Mirrors MolmoSpaces' reference `PI_Policy` (pi_policy.py): one third-person camera + wrist camera,
letterboxed to 224x224, 7 arm joints + normalized gripper as state, and an 8-step chunk of absolute
joint targets + binary gripper as output. Image/state conventions are shared with the RLDS converter
through ../molmospaces_utils.py so training and evaluation see identical inputs.

Runs inside the MolmoSpaces (`mlspaces`) env. Usage, with the openvla-oft repo root on PYTHONPATH:
    python -m molmo_spaces.evaluation.eval_main \
        experiments.robot.molmospaces.eval.molmospaces_oft_policy:OFTPolicyEvalConfig \
        --benchmark_dir <benchmark dir> --max_episodes 5 --no_wandb --output_dir <dir>
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import json_numpy
import numpy as np
import requests

from experiments.robot.molmospaces import molmospaces_utils as mu
from molmo_spaces.configs.policy_configs import BasePolicyConfig
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.evaluation.configs.evaluation_configs import JsonBenchmarkEvalConfig
from molmo_spaces.policy.base_policy import InferencePolicy, PolicyFactory, StatefulPolicy
from molmo_spaces.utils.function_utils import make_lenient

log = logging.getLogger(__name__)

SERVER_URL = os.environ.get("OFT_SERVER_URL", "http://localhost:8777")


@dataclass
class OFTPolicyState:
    actions_buffer: list[np.ndarray] | None = None
    current_buffer_index: int = 0
    starting_time: float | None = None


class OFTPolicy(InferencePolicy, StatefulPolicy):
    def __init__(self, exp_config, task=None) -> None:
        super().__init__(exp_config, task)
        policy_config = exp_config.policy_config
        self.server_url = policy_config.server_url
        self.checkpoint_path = policy_config.checkpoint_path
        self.camera_names = policy_config.camera_names
        self.chunk_size = policy_config.chunk_size
        self.grasping_threshold = policy_config.grasping_threshold
        self.session = None  # created lazily so the policy stays picklable for worker processes
        self.reset()

    def get_state(self):
        return OFTPolicyState(self.actions_buffer, self.current_buffer_index, self.starting_time)

    def set_state(self, state: OFTPolicyState):
        self.actions_buffer = state.actions_buffer
        self.current_buffer_index = state.current_buffer_index
        self.starting_time = state.starting_time

    def reset(self):
        self.actions_buffer = None
        self.current_buffer_index = 0
        self.starting_time = None

    def prepare_model(self):
        self.session = requests.Session()
        response = self.session.get(f"{self.server_url}/health", timeout=10)
        response.raise_for_status()
        log.info(f"Connected to OpenVLA-OFT server at {self.server_url}")

    def obs_to_model_input(self, obs):
        if isinstance(obs, (list, tuple)):
            obs = obs[0]
        exo_camera, wrist_camera = self.camera_names
        return {
            "full_image": mu.resize_with_pad(obs[exo_camera]),
            "wrist_image": mu.resize_with_pad(obs[wrist_camera]),
            "state": mu.qpos_to_state(obs["qpos"]),
            "instruction": self.task.get_task_description(),
        }

    def inference_model(self, model_input):
        if self.session is None:
            self.prepare_model()
        if self.starting_time is None:
            self.starting_time = time.time()
        if self.actions_buffer is None or self.current_buffer_index >= self.chunk_size:
            response = self.session.post(
                f"{self.server_url}/act", json={"encoded": json_numpy.dumps(model_input)}, timeout=120
            )
            if response.status_code != 200:
                raise RuntimeError(f"OpenVLA-OFT server error:\n{response.text}")
            self.actions_buffer = json_numpy.loads(response.json())
            self.current_buffer_index = 0
        model_output = self.actions_buffer[self.current_buffer_index]
        self.current_buffer_index += 1
        return model_output

    def model_output_to_action(self, model_output):
        close = model_output[7] > self.grasping_threshold
        return {
            "arm": np.asarray(model_output[:7], dtype=np.float64).reshape(7),
            "gripper": np.array([mu.GRIPPER_CMD_MAX if close else 0.0]),
        }

    def get_action_chunk(self, observation: Any) -> list[dict[str, np.ndarray]]:
        """Run one inference and return the whole chunk (same contract as PI_Policy.get_action_chunk)."""
        first_action = self.get_action(observation)
        buffered_outputs = self.actions_buffer[self.current_buffer_index : self.chunk_size]
        self.current_buffer_index = self.chunk_size
        return [first_action] + [self.model_output_to_action(out) for out in buffered_outputs]

    def get_info(self) -> dict:
        info = super().get_info()
        info["policy_name"] = "openvla-oft"
        info["policy_checkpoint"] = os.path.basename(str(self.checkpoint_path).rstrip("/"))
        info["policy_buffer_length"] = self.chunk_size
        info["policy_grasping_threshold"] = self.grasping_threshold
        info["policy_camera_names"] = list(self.camera_names)
        info["prompt"] = self.task.get_task_description()
        info["time_spent"] = time.time() - self.starting_time if self.starting_time else None
        info["timestamp"] = time.time()
        return json.loads(json.dumps(info, default=str))


class OFTPolicyConfig(BasePolicyConfig):
    server_url: str = SERVER_URL
    checkpoint_path: str = "openvla-oft"  # informational; the server owns the weights
    camera_names: list[str] = [mu.EXO_CAMERAS[0], mu.WRIST_CAMERA]  # [third-person, wrist]
    chunk_size: int = 8  # actions executed open-loop per inference (= NUM_ACTIONS_CHUNK)
    grasping_threshold: float = 0.5

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "learned"

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            self.policy_cls = OFTPolicy
            self.policy_factory = make_lenient(OFTPolicy)


class OFTPolicyEvalConfig(JsonBenchmarkEvalConfig):
    robot_config: FrankaRobotConfig = FrankaRobotConfig()
    policy_config: OFTPolicyConfig = OFTPolicyConfig()
    policy_dt_ms: float = 66.0  # 15 Hz, the MolmoBot-Data control rate
    end_on_success: bool = True

    def model_post_init(self, __context):
        super().model_post_init(__context)
        self.robot_config.action_noise_config.enabled = False
