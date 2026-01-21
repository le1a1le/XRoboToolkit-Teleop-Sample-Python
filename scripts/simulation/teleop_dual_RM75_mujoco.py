import os

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main(
    xml_path: str = os.path.join(ASSET_PATH, "dual_RM75/mult_scene_dual_RM75.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "dual_RM75/dual_rm75_robot.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = True,
):
    """
    启动双 Franka teleop 的示例脚本。
    注意：请确认 URDF 与 XML 中的 link/vis_target 名称匹配下面的 manipulator_config。
    """
    config = {
        "right_hand": {
            # 请把下面两个名称换成你 URDF / MuJoCo XML 中实际的末端执行器 link 和 vis target 名称
            "link_name": "right_tool0",    # Placo URDF 中的末端 link 名
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",  # MuJoCo scene XML 中的 mocap body 名
            # 如果需要 gripper 控制，取消注释并填写实际 joint 名和开闭值
            "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "right_trigger",
                "joint_names": ["right_dh_base_finger1_joint"],
                "open_pos": [0.04],
                "close_pos": [0.0],
            },
        },
        "left_hand": {
            "link_name": "left_tool0",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
            "vis_target": "left_target",
             "gripper_config": {
                "type": "parallel",
                "gripper_trigger": "left_trigger",
                "joint_names": ["left_dh_base_finger1_joint"],
                "open_pos": [0.04],
                "close_pos": [0.0],
            },
        },
    }

    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )

    # （可选）关节正则化任务
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)