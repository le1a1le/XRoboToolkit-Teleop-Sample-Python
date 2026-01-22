import os

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
import mujoco
import numpy as np


def main(
    xml_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/scene_dual_arm.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf"),
    scale_factor: float = 1.5,
    visualize_placo: bool = True,
    randomize_objects: bool = True, 
    table_center: tuple = (-0.5, 0, 0.775),
    table_size: tuple = (0.4, 0.4),
    z_offset: float = 0.01, 
):
    """
    Main function to run the dual UR5e teleoperation in MuJoCo.
    """
    config = {
        "right_hand": {
            "link_name": "right_tool0",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target",
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

    model = mujoco.MjModel.from_xml_path(xml_path)
   
    temp_data = mujoco.MjData(model)
    
    # 获取 home keyframe 的 qpos
    if model.nkey > 0:
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, temp_data, home_key_id)
            mj_qpos_init = temp_data.qpos.copy()  # 复制 home keyframe 的 qpos
        else:
            # 如果没有找到 home keyframe，使用默认值
            mj_qpos_init = np.zeros(model.nq)
    else:
        # 如果没有 keyframe，使用默认值
        mj_qpos_init = np.zeros(model.nq)


    # 找到 yuanzhu_joint0 的 qpos 索引
    yuanzhu_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yuanzhu_joint0")
    yuanzhu_qpos_addr = model.jnt_qposadr[yuanzhu_joint_id]

    # 设置位置 (x, y, z) 和四元数 (w, x, y, z)
    mj_qpos_init[yuanzhu_qpos_addr:yuanzhu_qpos_addr+3] = [-0.5, 0.2, 0.80]  # 位置
    mj_qpos_init[yuanzhu_qpos_addr+3:yuanzhu_qpos_addr+7] = [1, 0, 0, 0]  # 四元数 (w, x, y, z)

    # 同样设置 yuanzhukong
    yuanzhukong_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "yuanzhukong_joint0")
    yuanzhukong_qpos_addr = model.jnt_qposadr[yuanzhukong_joint_id]
    mj_qpos_init[yuanzhukong_qpos_addr:yuanzhukong_qpos_addr+3] = [-0.5, -0.2, 0.80]
    mj_qpos_init[yuanzhukong_qpos_addr+3:yuanzhukong_qpos_addr+7] = [1, 0, 0, 0]
  
    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
        mj_qpos_init=mj_qpos_init,
    )

    # additional constraints hardcoded here for now
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
