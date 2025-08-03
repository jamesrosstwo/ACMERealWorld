"""
Camera calibration from static grayscale images that contain an ArUco board.

The board:
    4 × 4 tags (ArUco 6×6, dict=50)
    Tag size        = 180 mm
    Gap between tags = 10 mm
Camera model: pinhole + plumb-bob
"""
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import logging
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# 1. Configuration – the only “global” variables
# --------------------------------------------------------------------------- #
BASE_OUT_DIR = Path("outputs").resolve().absolute()
OUT_DIR = BASE_OUT_DIR / datetime.now().strftime("calib_%Y%m%d_%H%M%S")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- PLOTTING CONFIG -------------------------------------------------- #
SAVE_DETECTION_PLOTS = False  # draw detected markers on images
MAX_DETECTION_PLOTS_PER_CAM = 5  # how many images to plot per camera
# --------------------------------------------------------------------------- #

# Board geometry
N_CAMS = 4
N_FRAMES = 1000
TAG_SIZE_MM = 180
GAP_MM = 10
TAGS_X = 4
TAGS_Y = 3
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
BOARD = cv2.aruco.GridBoard(
    size=(TAGS_X, TAGS_Y),
    markerLength=TAG_SIZE_MM / 1000.0,  # metres
    markerSeparation=GAP_MM / 1000.0,
    dictionary=ARUCO_DICT,
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUT_DIR / "calibrate.log"),
    ],
)


# --------------------------------------------------------------------------- #
# 2. Utilities
# --------------------------------------------------------------------------- #
def find_images(root: Path) -> dict[str, list[Path]]:
    """
    Return {camera_name: [sorted list of image paths]} under `root`.
    Assumes folder layout:  root/<camera_name>/<images>.png
    """
    cameras = {}
    for cam_dir in sorted(root.iterdir()):
        if not cam_dir.is_dir():
            continue
        images = sorted(cam_dir.glob("*.png"), key=lambda p: int(p.stem))
        print(f"Found {len(images)} images in {cam_dir}")
        if images:
            cameras[cam_dir.name] = images
    return cameras


# --------------------------------------------------------------------------- #
# 3. Plotting utilities (modular)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 3. Plotting utilities (modular)                                           #
# --------------------------------------------------------------------------- #
class DetectionPlotter:
    """
    Handles saving annotated images with detected markers.
    """

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir / f"detections"
        self.out_dir.mkdir(exist_ok=True)

    def plot(self,
             image: np.ndarray,
             corners,
             ids,
             cam_id: int,
             rejected,
             img_path: Path) -> None:
        """
        Draw:
            green rectangles for accepted markers
            red   rectangles for rejected candidates
        and save to disk.
        """
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Accepted markers
        cv2.aruco.drawDetectedMarkers(vis, corners, ids,
                                      borderColor=(0, 255, 0))  # green

        # Rejected candidates
        if rejected is not None and len(rejected) > 0:
            cv2.aruco.drawDetectedMarkers(vis, rejected, borderColor=(0, 0, 255))  # red

        out_name = self.out_dir / f"cam_{cam_id}_{img_path.stem}_detection.png"
        cv2.imwrite(str(out_name), vis)
        logging.debug(f"Saved detection plot: {out_name}")

    def skip(self) -> bool:
        return not SAVE_DETECTION_PLOTS


@dataclass(frozen=True)
class CamCalibData:
    timestamps: List[int]
    obj_points: List[np.ndarray]
    img_points: List[np.ndarray]
    validity: List[bool]


class CameraCalibrator:
    def __init__(self, base_path: Path, n_cams: int, n_frames: int):
        self._base_path = base_path
        self._n_frames = n_frames
        self._n_cams = n_cams
        self.plotter = DetectionPlotter(OUT_DIR)
        self.image_size = (1920, 1080)

        params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(ARUCO_DICT, params)

    def _detect_aruco(self, image: np.ndarray) -> tuple:
        corners, ids, rejected = self._detector.detectMarkers(image)
        return corners, ids, rejected

    def _cam_data(self, cam_path: Path, cam_id: int):
        im_paths = sorted(cam_path.glob("*.png"), key=lambda x: int(x.stem))
        timestamps = [int(x.stem) for x in im_paths]
        obj_points = []
        img_points = []
        validity = []
        for t, img_path in tqdm(zip(timestamps, im_paths), f"Images for camera {cam_id}", total=len(timestamps)):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

            valid = True
            if img is None:
                logging.warning(f"Could not read {img_path}")
                valid = False

            elif img.shape[::-1] != self.image_size:
                logging.warning(
                    f"Skipping {img_path} – size {img.shape[::-1]} differs from expected {self.image_size}"
                )
                valid = False

            corners, ids, rejected = self._detect_aruco(img)

            if not self.plotter.skip():
                self.plotter.plot(img, corners, ids, cam_id, rejected, img_path)
            if ids is None or len(ids) < 4:
                logging.debug(f"Not enough markers in {img_path}")
                valid = False

            if valid:
                obj_pts = []
                img_pts = []
                for c, i in zip(corners, ids.flatten()):
                    obj_pts.extend(BOARD.getObjPoints()[i])
                    img_pts.extend(c[0])
                obj_points.append(np.array(obj_pts, dtype=np.float32))
                img_points.append(np.array(img_pts, dtype=np.float32))
            else:
                obj_points.append(np.empty((0, 2)))
                img_points.append(np.empty((0, 3)))
            validity.append(valid)
        return CamCalibData(timestamps, obj_points, img_points, validity)

    def _calib_single(self, calib_data: CamCalibData):
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            objectPoints=[BOARD.getObjPoints()] * len(calib_data.obj_points),
            imagePoints=calib_data.img_points,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None,
            flags=cv2.CALIB_RATIONAL_MODEL,  # plumb-bob = 4 coeffs
        )
        return ret, K, D, rvecs, tvecs

    def calibrate(self) -> dict:
        """
        Run calibration and return a dictionary with results.
        """

        cam_data = dict()
        single_calibrations = dict()
        for cam_dir in tqdm(sorted(self._base_path.iterdir()), "Processing Cameras"):
            cam_id = int(cam_dir.stem[3:])
            cam_data[cam_id] = self._cam_data(cam_dir, cam_id)
            single_calibrations[cam_id] = self._calib_single(cam_data[cam_id])
        return

    def save_result(self, result: dict):
        """
        Save calibration result as YAML + human-readable txt.
        """
        out_file = OUT_DIR / f"{self.name}_calibration.yml"
        cv_file = cv2.FileStorage(str(out_file), cv2.FILE_STORAGE_WRITE)
        for k, v in result.items():
            if k in {"camera_matrix", "distortion_coeffs"}:
                cv_file.write(k, np.array(v))
            else:
                cv_file.write(k, v)
        cv_file.release()

        # Human-readable summary
        txt_file = OUT_DIR / f"{self.name}_summary.txt"
        with open(txt_file, "w") as f:
            f.write(f"Camera: {self.name}\n")
            f.write(f"Resolution: {result['image_width']} x {result['image_height']}\n")
            f.write(f"Images used: {result['num_images_used']}\n")
            f.write(f"Re-projection error: {result['reprojection_error']:.4f} px\n")
            f.write("Camera matrix (K):\n")
            for row in result["camera_matrix"]:
                f.write("  " + " ".join(f"{x:10.4f}" for x in row) + "\n")
            f.write("Distortion coefficients (k1 k2 p1 p2 k3):\n")
            f.write("  " + " ".join(f"{x:10.4f}" for x in result["distortion_coeffs"]))
            f.write("\n")
        logging.info(f"{self.name}: calibration saved to {out_file}")


def main(root: Path):
    """
    Calibrate every camera folder found under `root`.
    """
    cameras = find_images(root)
    if not cameras:
        logging.error("No camera folders found.")
        return

    cal = CameraCalibrator(root, N_CAMS, N_FRAMES)

    for name, images in cameras.items():
        logging.info(f"Starting calibration for camera '{name}'")
        result = cal.calibrate()
        if result:
            cal.save_result(result)


if __name__ == "__main__":
    main(Path(r"C:\Users\User\Desktop\ACMERealWorld\outputs\kalibr_postprocess_20250728_132842_bigboard"))
