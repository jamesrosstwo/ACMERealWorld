import hydra
from omegaconf import DictConfig
from client.nuc import NUCInterface
from client.realsense import RealSenseInterface
from client.teleop import MetaQuestInterface
from client.write import DataWriter


@hydra.main(config_path="config", config_name="collect")
def main(cfg: DictConfig):
    out_path = cfg.out_path
    nuc = NUCInterface(**cfg.nuc)
    realsense = RealSenseInterface(**cfg.realsense)
    quest = MetaQuestInterface(**cfg.quest)
    writer = DataWriter(**cfg.writer)

    for i in range(500):
        colors, depths = realsense.get_synchronized_frame()
        writer.write_frame(colors, depths)
        state = nuc.get_robot_state()
        eef_pos, gripper_force = quest.get_control()
        action = ...
        state.update(dict(action=action))
        writer.write_state(**state)
        nuc.send_control(eef_pos, gripper_force)
