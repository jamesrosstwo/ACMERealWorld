"""Episode postprocessing entry point.

Decodes RealSense ``.bag`` files recorded during collection into synchronized
multi-camera RGB video (MP4), IR stereo pairs (zarr), and aligned timestamps.
Optionally runs FoundationStereo to produce depth maps from the IR pairs.

When stereo depth is enabled, depth estimation runs asynchronously in background
threads (one per GPU) so that bagfile decoding and depth inference overlap.

Usage::

    python client/postprocess.py

Set ``stereo.enabled=true`` and provide a checkpoint path in
``config/postprocess.yaml`` to enable depth estimation.
"""
import logging
import queue
import shutil
import threading
from pathlib import Path
from typing import Callable, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from client.collect.realsense import RSBagProcessor
from client.collect.write import ACMEWriter

log = logging.getLogger(__name__)


def gather_data(
    episodes_path: str,
    n_frames: int,
    writer_cfg: DictConfig,
    realsense: DictConfig,
    on_episode_done: Optional[Callable[[Path], None]] = None,
) -> None:
    base_episodes_path = Path(episodes_path)
    base_episodes_path.mkdir(exist_ok=True, parents=True)

    ep_paths = sorted(base_episodes_path.iterdir(), key=lambda p: int(p.stem.split("_")[-1]))
    for ep_path in tqdm(ep_paths, "Postprocessing episodes"):
        try:
            completion_marker = ep_path / "COMPLETE"
            if completion_marker.exists():
                print(f"Skipping {ep_path}: already postprocessed")
            else:
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

            if on_episode_done is not None:
                on_episode_done(ep_path)
        except RuntimeError:
            print(f"bags unindexed for episode {ep_path}")


def _depth_worker(gpu_id: int, work_queue: queue.Queue, stereo_cfg: DictConfig):
    """Background worker: loads FoundationStereo on one GPU and processes episodes from the queue."""
    import torch
    from scripts.foundation_stereo import load_model, process_episode

    torch.cuda.set_device(gpu_id)
    torch.autograd.set_grad_enabled(False)
    model, model_args = load_model(
        stereo_cfg.ckpt_dir,
        scale=stereo_cfg.get("scale", 1.0),
        valid_iters=stereo_cfg.get("valid_iters", 32),
    )
    log.info("Depth worker ready on GPU %d", gpu_id)

    while True:
        ep_path = work_queue.get()
        if ep_path is None:
            break
        try:
            process_episode(ep_path, stereo_cfg.ckpt_dir, model=model, args=model_args)
        except Exception:
            log.exception("Depth processing failed for %s on GPU %d", ep_path, gpu_id)
        finally:
            work_queue.task_done()


@hydra.main(config_path="../config", config_name="postprocess", version_base=None)
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    stereo_enabled = cfg.get("stereo", {}).get("enabled", False)
    depth_queue: Optional[queue.Queue] = None
    workers: list[threading.Thread] = []

    if stereo_enabled:
        import torch

        num_gpus = torch.cuda.device_count()
        depth_queue = queue.Queue()
        for gpu_id in range(num_gpus):
            t = threading.Thread(
                target=_depth_worker,
                args=(gpu_id, depth_queue, cfg.stereo),
                daemon=True,
            )
            t.start()
            workers.append(t)
        log.info("Started %d depth worker(s) across %d GPU(s)", num_gpus, num_gpus)

    gather_data(
        cfg.episodes_path,
        cfg.max_episode_timesteps,
        cfg.writer,
        cfg.realsense,
        on_episode_done=depth_queue.put if depth_queue else None,
    )

    if stereo_enabled:
        print("Waiting for background depth processing to finish...")
        depth_queue.join()
        for _ in workers:
            depth_queue.put(None)
        for t in workers:
            t.join()


if __name__ == "__main__":
    main()
