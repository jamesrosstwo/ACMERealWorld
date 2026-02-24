"""Episode postprocessing entry point.

Decodes RealSense ``.bag`` files recorded during collection into synchronized
multi-camera RGB video (MP4), IR stereo pairs (zarr), and aligned timestamps.
Optionally runs FoundationStereo to produce depth maps from the IR pairs.

Usage::

    python client/postprocess.py

Set ``stereo.enabled=true`` and provide a checkpoint path in
``config/postprocess.yaml`` to enable depth estimation.
"""
import shutil
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from client.collect.realsense import RSBagProcessor
from client.collect.write import ACMEWriter


def gather_data(episodes_path: str, n_frames: int, writer_cfg: DictConfig, realsense: DictConfig) -> Path:
    base_episodes_path = Path(episodes_path)
    base_episodes_path.mkdir(exist_ok=True, parents=True)

    ep_paths = sorted(base_episodes_path.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))
    for ep_path in tqdm(ep_paths, "Postprocessing episodes"):
        try:
            completion_marker = ep_path / "COMPLETE"
            if completion_marker.exists():
                print(f"Skipping {ep_path}: already postprocessed")
                continue
            print(f"Processing episode {ep_path}")
            try:
                shutil.rmtree(Path(ep_path / "captures"))
            except FileNotFoundError:
                pass
            bagpaths = sorted(Path(ep_path).glob("*.bag"))
            bagpaths = [p for p in bagpaths if not p.stem.endswith(".orig")]
            serials = [p.stem for p in bagpaths]
            writer = ACMEWriter(ep_path, serials=serials, **writer_cfg)
            rs_interface = RSBagProcessor(bagpaths, **realsense)
            print(f"found {len(bagpaths)} bags: {serials}")
            for color, color_tmstmp, ir_left, ir_right, serial in tqdm(rs_interface.process_all_frames()):
                try:
                    writer.write_capture_frame(serial, color_tmstmp, color, ir_left, ir_right)
                except IndexError:
                    continue
            writer.flush()
            completion_marker.touch()
        except RuntimeError as e:
            print(f"bags unindexed for episode {ep_path}")


def run_stereo_depth(episodes_path: str, stereo_cfg: DictConfig):
    """Run FoundationStereo on all completed episodes to produce depth.zarr."""
    import torch
    from scripts.foundation_stereo import load_model, process_episode

    torch.autograd.set_grad_enabled(False)
    model, model_args = load_model(
        stereo_cfg.ckpt_dir,
        scale=stereo_cfg.get("scale", 1.0),
        valid_iters=stereo_cfg.get("valid_iters", 32),
    )

    base = Path(episodes_path)
    ep_paths = sorted(base.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))
    for ep_path in tqdm(ep_paths, "FoundationStereo depth"):
        if not (ep_path / "COMPLETE").exists():
            continue
        process_episode(ep_path, stereo_cfg.ckpt_dir, model=model, args=model_args)


@hydra.main(config_path="../config", config_name="postprocess")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    n_frames = cfg.max_episode_timesteps
    gather_data(cfg.episodes_path, n_frames, cfg.writer, cfg.realsense)

    if cfg.get("stereo", {}).get("enabled", False):
        run_stereo_depth(cfg.episodes_path, cfg.stereo)


if __name__ == "__main__":
    main()
