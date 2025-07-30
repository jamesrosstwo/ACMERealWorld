import shutil
from pathlib import Path
from datetime import datetime
import cv2


def convert_to_kalibr_format(source_root: Path, camera_count: int):
    base_out_dir = Path("outputs").resolve().absolute()
    dest_root = base_out_dir / datetime.now().strftime("kalibr_postprocess_%Y%m%d_%H%M%S")
    dest_root.mkdir(parents=True)

    for cam_idx in range(camera_count):
        cam_src = source_root / f"cam_{cam_idx}"
        cam_dest = dest_root / f"cam{cam_idx}"
        cam_dest.mkdir(parents=True, exist_ok=True)

        shutil.copyfile("target.yaml", cam_dest / "target.yaml")

        for img_file in sorted(cam_src.glob("*.png")):
            stem = img_file.stem  # e.g., '1234567890'
            try:
                timestamp_ms = int(stem)
                timestamp_us = timestamp_ms * 1000000  # Convert ms to µs
                new_filename = f"{timestamp_us}.png"
            except ValueError:
                print(f"Skipping file with non-numeric name: {img_file.name}")
                continue

            dest_img_path = cam_dest / new_filename

            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read image: {img_file}")
                continue

            cv2.imwrite(str(dest_img_path), img)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert RealSense image dataset to Kalibr format (grayscale)")
    parser.add_argument("source", type=str, help="Path to original output directory (with cam_0/, cam_1/...)")
    parser.add_argument("--cams", type=int, default=4, help="Number of cameras")
    args = parser.parse_args()

    convert_to_kalibr_format(Path(args.source), args.cams)
