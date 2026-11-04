"""The models the DroidAdapter serves (`python -m roboarc.vla.droid_adapter --model NAME`): DROID-trained policies in
DROID's units. Each runs in its own Python environment (scripts/vla_droid.sh picks it); a model's libraries are
imported in load(), never when this module is.

    jointpos    openpi's pi05_droid_jointpos: absolute joint positions + the gripper position (0 = open .. 1 = closed),
                the `droid` convention                                                      (openpi's environment)
    velocity    openpi's pi05_droid: DROID's normalised joint velocities, the `droid-velocity` convention
    polaris     pi05_droid_jointpos fine-tuned with PolaRiS' simulated data: `droid`
    molmoact2   allenai/MolmoAct2-DROID in float32 (the model card's reported runs, about 24 GB of GPU memory):
                absolute joint positions + the gripper position, `droid`           (allenai/molmoact2's environment)
    molmoact2-bf16   the same in bfloat16 (the model card's smaller path, under 16 GB)
    hold        stub_server's HoldPolicy instead of a model (numpy only), for checking a setup: `droid`

A model gets the droid convention's observation with DROID's q7 (the adapter's wrist offset applied, no "roboarc"
entry) and returns {"actions": (H, 8)} in DROID's units."""
from __future__ import annotations

import numpy as np

from .stub_server import HoldPolicy

JOINTS = "observation/joint_position"
GRIPPER = "observation/gripper_position"
IMAGES = ("observation/exterior_image_1_left", "observation/exterior_image_2_left", "observation/wrist_image_left")


class DroidModel:
    """A DROID-trained policy: load() once before serving, then infer(obs) -> {"actions": (H, 8)}. A subclass loads its
    weights in _load() and sets self.policy (anything with infer), or overrides infer."""
    name = "?"
    velocity = False      # the actions are DROID's joint velocities (the droid-velocity convention)
    image_size = 224      # the images' size the model is run with (the runner's --image-size)

    def load(self) -> None:
        """The weights, then one call on a blank observation: the first call compiles (JAX) or sets up CUDA for longer
        than the clients' 20 s keepalive, and the server answers no ping while a call runs."""
        self._load()
        blank = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        self.infer({**{k: blank for k in IMAGES}, JOINTS: np.zeros(7), GRIPPER: np.zeros(1), "prompt": ""})

    def _load(self) -> None:
        raise NotImplementedError

    def infer(self, obs: dict) -> dict:
        return self.policy.infer(obs)


class HoldModel(DroidModel):
    name = "roboarc stub (hold)"

    def _load(self) -> None:
        self.policy = HoldPolicy(horizon=15)       # one DROID chunk


class OpenpiModel(DroidModel):
    """An openpi checkpoint served by openpi's own policy (its input and output transforms): `config` is openpi's
    training config, `checkpoint` its directory (gs:// paths download into ~/.cache/openpi or $OPENPI_DATA_HOME)."""

    def __init__(self, config: str, checkpoint: str, velocity: bool = False):
        self.config, self.checkpoint, self.velocity = config, checkpoint, velocity
        self.name = f"{config} ({checkpoint})"

    def _load(self) -> None:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config
        self.policy = policy_config.create_trained_policy(openpi_config.get_config(self.config), self.checkpoint)


class MolmoAct2Model(DroidModel):
    """allenai/MolmoAct2-DROID (5B) through its Hugging Face code, called as its model card and allenai/molmoact2's
    examples/droid/host_server_droid.py call it: continuous actions from 10 flow steps, no depth reasoning, the prompt
    normalised by the model; `dtype` float32, or bfloat16 weights under bfloat16 autocast (the model card's two paths).
    It gets two images, exterior 1 and the wrist, as that server and allenai/molmoact2's simulated and real DROID clients
    send them (scripts/vla_droid.sh renders agentview, aimed as in that simulation, as exterior 1), each cut back to
    DROID's 16:9 frame: 320 x 180 when the runner pads the 848 x 480 cameras to 320 x 320 (--image-size 320). It returns
    15 actions: absolute joint positions (the model clips its state and actions to DROID's 1-99 % range) and a
    continuous gripper position, clipped here to [0, 1]."""
    repo, revision = "allenai/MolmoAct2-DROID", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"
    image_size = 320

    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        self.name = f"{self.repo} ({self.revision[:7]}, {dtype})"

    def _load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor
        path = snapshot_download(self.repo, revision=self.revision)   # local: the model reads its norm_stats.json there
        # extra_special_tokens={}: the tokenizer config's list breaks transformers 4.57 (as the official server does)
        self.processor = AutoProcessor.from_pretrained(path, trust_remote_code=True, extra_special_tokens={})
        self.model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True,
                                                                 dtype=getattr(torch, self.dtype))
        self.model = self.model.to("cuda").eval()
        self.generator = torch.Generator("cuda").manual_seed(0)

    def infer(self, obs: dict) -> dict:
        import torch
        images, state = molmoact2_inputs(obs)
        bf16 = torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.dtype == "bfloat16")
        with torch.inference_mode(), bf16:
            out = self.model.predict_action(
                processor=self.processor, images=images, task=obs["prompt"], state=state, norm_tag="franka_droid",
                inference_action_mode="continuous", enable_depth_reasoning=False, num_steps=10,
                generator=self.generator, normalize_language=True, enable_cuda_graph=False)
        return {"actions": molmoact2_actions(out.actions)}


def droid_frame(image: np.ndarray) -> np.ndarray:
    """A square image the runner padded from a 16:9 camera (resize_with_pad) -> its middle 9/16 of the rows: DROID's
    frame shape (320 x 180 from 320 x 320)."""
    h = round(image.shape[1] * 9 / 16)
    top = (image.shape[0] - h) // 2
    return np.ascontiguousarray(image[top:top + h])


def molmoact2_inputs(obs: dict) -> tuple[list, np.ndarray]:
    """MolmoAct2-DROID's inputs: the exterior 1 and wrist frames and the state (8,) float32: DROID's 7 joints and the
    gripper position (0 = open .. 1 = closed)."""
    images = [droid_frame(np.asarray(obs[k], dtype=np.uint8)) for k in (IMAGES[0], IMAGES[2])]
    state = np.r_[np.asarray(obs[JOINTS]).reshape(7), np.asarray(obs[GRIPPER]).reshape(1)].astype(np.float32)
    return images, state


def molmoact2_actions(actions) -> np.ndarray:
    """The model's (1, H, 8) actions (a tensor or an array) -> (H, 8) float64, the gripper clipped to [0, 1]."""
    if hasattr(actions, "detach"):
        actions = actions.detach().float().cpu().numpy()
    a = np.array(actions, dtype=np.float64).reshape(-1, 8)
    a[:, 7] = np.clip(a[:, 7], 0.0, 1.0)
    return a


# The joint-position checkpoints take openpi's pi05_droid_jointpos_polaris config: DROID's joint-position inputs and
# absolute outputs, the norm stats from the checkpoint.
MODELS = {"jointpos": OpenpiModel("pi05_droid_jointpos_polaris", "gs://openpi-assets/checkpoints/pi05_droid_jointpos"),
          "velocity": OpenpiModel("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid", velocity=True),
          "polaris": OpenpiModel("pi05_droid_jointpos_polaris",
                                 "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris"),
          "molmoact2": MolmoAct2Model("float32"),
          "molmoact2-bf16": MolmoAct2Model("bfloat16"),
          "hold": HoldModel()}
