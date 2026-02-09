"""Policy evaluation subpackage.

Runs a learned policy on the robot using RealSense camera observations.
Coordinates inference via an HTTP policy server, sends resulting actions
to the Franka Panda, and records evaluation episodes with trajectory
visualizations.
"""
