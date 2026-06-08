"""Bench-test the Robotiq 2F-85 gripper backend in isolation (arm idle).

Constructs the same :class:`~client.nuc.RobotiqGripper` used by the collect/eval
loops, activates and calibrates the gripper, then drives a few open/close cycles
through the real ``act_async`` hysteresis dispatch. After each command it prints
the ``gripper_force`` observation exactly as ``NUCInterface.get_robot_state``
computes it (``1 - width/max_width``) so you can confirm the serial link,
activation, and obs mapping (~0.0 open, ~1.0 closed) before running the full
pipeline.

Usage::

    python -m scripts.test_robotiq                      # auto-detect port
    python -m scripts.test_robotiq --port /dev/ttyUSB0
    python -m scripts.test_robotiq --force 80 --speed 200 --cycles 3
"""
import argparse
import time

from client.nuc import RobotiqGripper


def _gripper_force(state) -> float:
    """Mirror NUCInterface.get_robot_state's gripper_force computation."""
    return 1.0 - state.width / state.max_width


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="auto",
                    help="serial port / by-id symlink, or 'auto' (default)")
    ap.add_argument("--device-id", type=int, default=9)
    ap.add_argument("--speed", type=int, default=255, help="0-255")
    ap.add_argument("--force", type=int, default=100, help="0-255")
    ap.add_argument("--max-width-m", type=float, default=0.085)
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--dwell", type=float, default=2.0,
                    help="seconds to wait for travel + settle between commands")
    args = ap.parse_args()

    print(f"Connecting to Robotiq on port={args.port} (device_id={args.device_id}) ...")
    gripper = RobotiqGripper(
        port=args.port,
        device_id=args.device_id,
        speed=args.speed,
        force=args.force,
        max_width_m=args.max_width_m,
        hysteresis=0.1,
        logging=True,
    )

    def report(label: str):
        # Let the async move thread finish before reading.
        time.sleep(args.dwell)
        state = gripper.get_state()
        print(f"  [{label}] width={state.width*1000:6.1f}mm  "
              f"gripper_force={_gripper_force(state):.3f}  "
              f"(0.0=open, 1.0=closed)")

    report("initial")
    for i in range(args.cycles):
        print(f"Cycle {i+1}/{args.cycles}")
        # Drive through the same hysteresis dispatch the control loops use:
        # 1.0 -> grasp, 0.0 -> open. act_async spawns the move on a thread.
        gripper.act_async(1.0)
        report("grasp")
        gripper.act_async(0.0)
        report("open")

    print("Done.")


if __name__ == "__main__":
    main()
