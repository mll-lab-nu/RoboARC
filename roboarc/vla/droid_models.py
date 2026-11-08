"""The models the DroidAdapter serves (`python -m roboarc.vla.droid_adapter --model NAME`): DROID-trained policies in
DROID's units. Each runs in its own Python environment (scripts/vla_droid.sh picks it); a model's libraries are
imported in load(), never when this module is.

    jointpos    openpi's pi05_droid_jointpos: absolute joint positions + the gripper position (0 = open .. 1 = closed,
                binarised at 0.5), the `droid` convention                                   (openpi's environment)
    velocity    openpi's pi05_droid: DROID's normalised joint velocities, the `droid-velocity` convention
    polaris     pi05_droid_jointpos fine-tuned with PolaRiS' simulated data: `droid`
    molmoact2   allenai/MolmoAct2-DROID in float32 (the model card's reported runs, about 24 GB of GPU memory):
                absolute joint positions + a continuous gripper position (its DROID client does not binarise it),
                `droid`                                                            (allenai/molmoact2's environment)
    molmoact2-bf16   the same in bfloat16 (the model card's smaller path, under 16 GB)
    groot       nvidia/GR00T-N1.7-DROID in bfloat16: absolute joint positions + the gripper position (binarised at 0.5),
                `droid`                                                         (NVIDIA/Isaac-GR00T's environment)
    cosmos3     nvidia/Cosmos3-Nano-Policy-DROID: absolute joint positions + the gripper position (binarised at 0.5),
                `droid`, three images                                       (NVIDIA/cosmos-framework's environment)
    hold        stub_server's HoldPolicy instead of a model (numpy only), for checking a setup: `droid`

A model gets the droid convention's observation with DROID's q7 (the adapter's wrist offset applied, no "roboarc"
entry) and returns {"actions": (H, 8)} in DROID's units. Its pinned versions (`version`) go into the server's metadata,
and from there into every episode's live.json and result.json."""
from __future__ import annotations

import numpy as np

from .stub_server import HoldPolicy

JOINTS = "observation/joint_position"
GRIPPER = "observation/gripper_position"
IMAGES = ("observation/exterior_image_1_left", "observation/exterior_image_2_left", "observation/wrist_image_left")
OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"     # the openpi the pi0.5 checkpoints are served with
# Franka's kinematics (modified DH: a, d, alpha per joint), then the flange 0.107 m along the last axis: panda_link8
FRANKA_DH = ((0, 0.333, 0), (0, 0, -np.pi / 2), (0, 0.316, np.pi / 2), (0.0825, 0, np.pi / 2),
             (-0.0825, 0.384, -np.pi / 2), (0, 0, np.pi / 2), (0.088, 0, np.pi / 2))
FLANGE = 0.107
DROID_EEF_ROTATION_CORRECT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)  # Isaac-GR00T's droid_frame.py


class DroidModel:
    """A DROID-trained policy: load() once before serving, then infer(obs) -> {"actions": (H, 8)}. A subclass loads its
    weights in _load() and sets self.policy (anything with infer), or overrides infer."""
    name = "?"
    version: dict = {}    # the pinned code and weights
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
    training config, `checkpoint` its directory (gs:// paths download into ~/.cache/openpi or $OPENPI_DATA_HOME) and
    `files_sha256` its version: the sha256 of its files' sorted "<path> <md5>" lines, with the MD5s (base64) Google Cloud
    Storage lists (https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/<name>/). The
    gripper actions are binarised at 0.5, as openpi's DROID client (examples/droid/main.py) and sim-evals' execute
    them."""

    def __init__(self, config: str, checkpoint: str, files_sha256: str, velocity: bool = False):
        self.config, self.checkpoint, self.velocity = config, checkpoint, velocity
        self.name = f"{config} ({checkpoint})"
        self.version = {"openpi": OPENPI_COMMIT, "checkpoint": checkpoint, "checkpoint_files_sha256": files_sha256}

    def _load(self) -> None:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config
        self.policy = policy_config.create_trained_policy(openpi_config.get_config(self.config), self.checkpoint)

    def infer(self, obs: dict) -> dict:
        out = self.policy.infer(obs)
        a = np.array(out["actions"])
        a[:, 7] = a[:, 7] > 0.5
        return {**out, "actions": a}


class MolmoAct2Model(DroidModel):
    """allenai/MolmoAct2-DROID (5B) through its Hugging Face code, called as its model card and allenai/molmoact2's
    examples/droid/host_server_droid.py call it: continuous actions from 10 flow steps, no depth reasoning, the prompt
    normalised by the model; `dtype` float32, or bfloat16 weights under bfloat16 autocast (the model card's two paths).
    It gets two images, exterior 1 and the wrist, as that server and allenai/molmoact2's simulated and real DROID clients
    send them (scripts/vla_droid.sh gives agentview, aimed as in that simulation, as exterior 1), each cut back to
    DROID's 16:9 frame: 320 x 180 when the runner pads the 848 x 480 cameras to 320 x 320 (--image-size 320). It returns
    15 actions: absolute joint positions (the model clips its state and actions to DROID's 1-99 % range) and a
    continuous gripper position, clipped here to [0, 1]."""
    repo, revision = "allenai/MolmoAct2-DROID", "d8c1abd8a27d8e859455bbe514df2bcc617db0fb"
    image_size = 320

    def __init__(self, dtype: str = "float32"):
        self.dtype = dtype
        self.name = f"{self.repo} ({self.revision[:7]}, {dtype})"
        self.version = {"model": self.repo, "revision": self.revision}

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


class Gr00tModel(DroidModel):
    """nvidia/GR00T-N1.7-DROID (3B, its backbone nvidia/Cosmos-Reason2-2B) through NVIDIA/Isaac-GR00T's Gr00tPolicy in
    bfloat16, asked as that repository's DROID client (examples/DROID/main_gr00t.py) asks it (groot_request): its own
    4 flow steps; 40 actions back, absolute joint positions and the gripper position, binarised at 0.5 as that client
    executes them (groot_actions); that client executes 15 per query (scripts/vla_droid.sh: --horizon 15)."""
    repo, revision = "nvidia/GR00T-N1.7-DROID", "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5"
    code = "1a1837f20538b7d7e21f977a11a5aee14f99803c"     # NVIDIA/Isaac-GR00T
    backbone = "nvidia/Cosmos-Reason2-2B"                   # loaded by the model from its Hugging Face main
    image_size = 320

    def __init__(self):
        self.name = f"{self.repo} ({self.revision[:7]}, bfloat16)"
        self.version = {"isaac_gr00t": self.code, "model": self.repo, "revision": self.revision}

    def _load(self) -> None:
        import torch
        from huggingface_hub import snapshot_download
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        self.version["backbone"] = f"{self.backbone}@{snapshot_download(self.backbone).rsplit('/', 1)[-1]}"
        torch.manual_seed(0)
        self.policy = Gr00tPolicy("OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT",
                                  snapshot_download(self.repo, revision=self.revision), device="cuda")

    def infer(self, obs: dict) -> dict:
        out, _ = self.policy.get_action(groot_request(obs))
        return {"actions": groot_actions(out)}


class Cosmos3Model(DroidModel):
    """nvidia/Cosmos3-Nano-Policy-DROID (16B) through NVIDIA/cosmos-framework's RoboLab policy service, asked as
    RoboLab's Cosmos3 client (policies/cosmos3) asks it (cosmos3_request): three DROID frames at 640 x 360, the wrist
    and both exterior cameras (the service puts them in one 640 x 540 image, the exteriors at half size below the
    wrist, as that client does; scripts/vla_droid.sh gives front_oblique, on the robot's right, as exterior 2), the
    joints and the gripper position (the service flips the gripper both ways); the service's defaults: 4 UniPC steps,
    guidance 3, a seed per query drawn from seed 0. 32 actions back, absolute joint positions and the gripper position,
    binarised at 0.5 as that client executes them (cosmos3_actions); that client executes all 32 (--horizon 32)."""
    repo, revision = "nvidia/Cosmos3-Nano-Policy-DROID", "24aa5d14a6cbef19ac6930bd2bf67c9c08e56be0"
    code = "bbd4a67c3cf54cc6699464ad1259f3a0f6537da9"     # NVIDIA/cosmos-framework
    image_size = 640

    def __init__(self):
        self.name = f"{self.repo} ({self.revision[:7]})"
        self.version = {"cosmos_framework": self.code, "model": self.repo, "revision": self.revision}

    def _load(self) -> None:
        from cosmos_framework.scripts.action_policy_server_robolab import RobolabPolicyService, RobolabServerArgs
        self.policy = RobolabPolicyService(RobolabServerArgs(checkpoint_path=self.repo, hf_revision=self.revision))

    def infer(self, obs: dict) -> dict:
        return {"actions": cosmos3_actions(self.policy.infer(cosmos3_request(obs))["action"])}


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


def droid_eef_9d(q) -> np.ndarray:
    """GR00T-N1.7-DROID's eef_9d state (9,) from DROID's joints: DROID's cartesian position (panda_link8 in the base
    frame, its rotation as DROID's extrinsic "xyz" Euler angles) converted as Isaac-GR00T's compute_eef_9d converts it:
    the position, then the angles read as scipy's "XYZ", times its rotation correction, the first two rows. It equals
    the eef_9d of Isaac-GR00T's demo_data/droid_sample from that data's joints (to 3e-7)."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4)
    for (a, d, alpha), theta in zip(FRANKA_DH + ((0, FLANGE, 0),), [*np.asarray(q, float).reshape(7), 0.0]):
        ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    euler = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    rot = Rotation.from_euler("XYZ", euler).as_matrix() @ DROID_EEF_ROTATION_CORRECT
    return np.r_[T[:3, 3], rot[:2].reshape(6)]


def groot_request(obs: dict) -> dict:
    """Gr00tPolicy's request (batch 1, one time step: the checkpoint's delta indices) from a droid-convention
    observation, as Isaac-GR00T's DROID client builds it: the exterior 1 and wrist frames (DROID's 320 x 180,
    droid_frame), the state (eef_9d, the gripper position, the joints; float32) and the prompt."""
    q = np.asarray(obs[JOINTS], dtype=np.float64).reshape(7)
    state = {"eef_9d": droid_eef_9d(q), "gripper_position": np.asarray(obs[GRIPPER]).reshape(1), "joint_position": q}
    return {"video": {k: droid_frame(np.asarray(obs[IMAGES[i]], dtype=np.uint8))[None, None]
                      for k, i in (("exterior_image_1_left", 0), ("wrist_image_left", 2))},
            "state": {k: v.astype(np.float32)[None, None] for k, v in state.items()},
            "language": {"annotation.language.language_instruction": [[obs["prompt"]]]}}


def groot_actions(out: dict) -> np.ndarray:
    """Gr00tPolicy's actions (batch 1) -> (H, 8) float64: the absolute joint positions and the gripper position,
    binarised at 0.5."""
    a = np.concatenate([out["joint_position"][0], out["gripper_position"][0]], axis=1).astype(np.float64)
    a[:, 7] = a[:, 7] > 0.5
    return a


def cosmos3_request(obs: dict) -> dict:
    """The RoboLab policy service's request from a droid-convention observation: the three images cut back to DROID's
    16:9 frame (640 x 360 from the runner's 640 x 640), DROID's joints and gripper position as they are."""
    return {**{k: droid_frame(np.asarray(obs[k], dtype=np.uint8)) for k in IMAGES},
            JOINTS: np.asarray(obs[JOINTS], dtype=np.float32).reshape(7),
            GRIPPER: np.asarray(obs[GRIPPER], dtype=np.float32).reshape(1), "prompt": obs["prompt"]}


def cosmos3_actions(action) -> np.ndarray:
    """The service's (H, 8) actions -> (H, 8) float64, the gripper position binarised at 0.5."""
    a = np.array(action, dtype=np.float64)
    a[:, 7] = a[:, 7] > 0.5
    return a


# The joint-position checkpoints take openpi's pi05_droid_jointpos_polaris config: DROID's joint-position inputs and
# absolute outputs, the norm stats from the checkpoint. Their files were uploaded on 2025-12-13 (pi05_droid 2025-09-08).
MODELS = {"jointpos": OpenpiModel("pi05_droid_jointpos_polaris", "gs://openpi-assets/checkpoints/pi05_droid_jointpos",
                                  "1062e5b3dfecce4a0f6e8dbaf0793820c62e6cb32dc86096f9cc84664764cb50"),
          "velocity": OpenpiModel("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid",
                                  "92d6cfebe9293bcdbd23afdb17f3fdc147cd4c63bcb5f6c186e0dcccbf21a73a", velocity=True),
          "polaris": OpenpiModel("pi05_droid_jointpos_polaris",
                                 "gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris",
                                 "5cbf57c52f52e02e1d8cc27945b46ff2bb87e051d97a43fed7e11bc6c5bf4faa"),
          "molmoact2": MolmoAct2Model("float32"),
          "molmoact2-bf16": MolmoAct2Model("bfloat16"),
          "groot": Gr00tModel(),
          "cosmos3": Cosmos3Model(),
          "hold": HoldModel()}
