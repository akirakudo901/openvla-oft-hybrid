"""
oft_policy_server.py

Serves an OpenVLA-OFT checkpoint fine-tuned on MolmoBot-Data over HTTP, for MolmoSpaces benchmark
evaluation (the simulator runs in a separate env with a different torch version).

The checkpoint is a LoRA run directory (merge_lora_during_training=False): lora_adapter/, action head,
proprio projector, processor files and dataset_statistics.json. The adapter is merged into the base
OpenVLA-7B weights in GPU memory, so no merged 15 GB copy is written to disk.

POST /act  {"encoded": json_numpy.dumps({"full_image", "wrist_image", "state", "instruction"})}
  -> json_numpy.dumps(list of NUM_ACTIONS_CHUNK actions, each (ACTION_DIM,), un-normalized)

Usage (in the `oft` env, repo root on PYTHONPATH; "molmo" in the path selects the MOLMOSPACES constants):
    python experiments/robot/molmospaces/eval/oft_policy_server.py \
        --base_checkpoint $WS/models/openvla-7b \
        --pretrained_checkpoint <run dir> --unnorm_key molmobot_pick_dev
"""

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import draccus
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from peft import PeftModel

from experiments.robot.openvla_utils import (
    DEVICE,
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


@dataclass
class ServerConfig:
    # fmt: off
    host: str = "0.0.0.0"
    port: int = 8777

    base_checkpoint: Union[str, Path] = ""           # Base OpenVLA-7B weights (local dir)
    pretrained_checkpoint: Union[str, Path] = ""     # Fine-tuning run dir with lora_adapter/ and component checkpoints
    unnorm_key: str = "molmobot_pick_dev"            # Dataset name used for action/proprio (un)normalization

    use_l1_regression: bool = True
    use_diffusion: bool = False
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = True
    center_crop: bool = True                         # Trained with random-crop image aug -> center crop at test time
    lora_rank: int = 32

    model_family: str = "openvla"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    # fmt: on


class OFTPolicyServer:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        run_dir = os.path.expanduser(str(cfg.pretrained_checkpoint))

        # get_vla loads whatever `pretrained_checkpoint` points to: load the base model first ...
        cfg.pretrained_checkpoint = os.path.expanduser(str(cfg.base_checkpoint))
        vla = get_vla(cfg)
        # ... then merge the fine-tuned LoRA adapter into it in memory
        vla = PeftModel.from_pretrained(vla, os.path.join(run_dir, "lora_adapter")).to(DEVICE)
        self.vla = vla.merge_and_unload().eval()

        # Everything else comes from the fine-tuning run dir
        cfg.pretrained_checkpoint = run_dir
        with open(os.path.join(run_dir, "dataset_statistics.json")) as f:
            self.vla.norm_stats = json.load(f)
        assert cfg.unnorm_key in self.vla.norm_stats, f"{cfg.unnorm_key} not in {list(self.vla.norm_stats)}"
        self.proprio_projector = get_proprio_projector(cfg, self.vla.llm_dim, PROPRIO_DIM) if cfg.use_proprio else None
        self.action_head = get_action_head(cfg, self.vla.llm_dim)
        self.processor = get_processor(cfg)
        print(f"[server] loaded {run_dir} (chunk {NUM_ACTIONS_CHUNK}, proprio {PROPRIO_DIM})")

    def act(self, payload: Dict[str, Any]):
        try:
            observation = json.loads(payload["encoded"])
            actions = get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                observation,
                observation["instruction"],
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                use_film=self.cfg.use_film,
            )
            return JSONResponse(json_numpy.dumps([a.astype("float32") for a in actions]))
        except Exception:  # keep serving; the client sees the error and fails its episode loudly
            logging.error(traceback.format_exc())
            return JSONResponse({"error": traceback.format_exc()}, status_code=500)

    def run(self) -> None:
        app = FastAPI()
        app.post("/act")(self.act)
        app.get("/health")(lambda: {"ok": True})
        uvicorn.run(app, host=self.cfg.host, port=self.cfg.port)


@draccus.wrap()
def main(cfg: ServerConfig) -> None:
    OFTPolicyServer(cfg).run()


if __name__ == "__main__":
    main()
