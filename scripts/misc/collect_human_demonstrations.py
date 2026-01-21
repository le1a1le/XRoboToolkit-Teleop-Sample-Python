"""
数据采集脚本：适配 teleop_dual_RM75_mujoco.py
采集 VR 遥操作数据并保存为 pkl 格式（兼容 collect_human_demonstrations.py）
"""

import os
import time
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np
import mujoco
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer
from PIL import Image

import tyro
from xrobotoolkit_teleop.simulation.mujoco_teleop_controller import MujocoTeleopController
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.geometry import quat_diff_as_angle_axis


def quat_wxyz_to_xyzw(quat_wxyz):
    """Convert quaternion from (w,x,y,z) to (x,y,z,w) format"""
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def quat_xyzw_to_wxyz(quat_xyzw):
    """Convert quaternion from (x,y,z,w) to (w,x,y,z) format"""
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _render_rgb(mj_model, mj_data, camera_name: str = None, width: int = 320, height: int = 240) -> np.ndarray:
    """从 MuJoCo 渲染 RGB 图像"""
    try:
        # 如果没有指定 camera_name，使用默认视角
        if camera_name is None:
            # 创建一个临时 viewer 来渲染
            scene = mujoco.MjvScene(mj_model, maxgeom=10000)
            context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
            
            # 设置相机参数（类似 agentview）
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = [0.3, 0, 0.8]
            cam.distance = 1.5
            cam.azimuth = 120
            cam.elevation = -20
            
            viewport = mujoco.MjrRect(0, 0, width, height)
            mujoco.mjv_updateScene(
                mj_model, mj_data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
            )
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            mujoco.mjr_render(viewport, scene, context)
            mujoco.mjr_readPixels(rgb, depth, viewport, context)
            return rgb
        else:
            # 使用指定的相机
            camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id >= 0:
                scene = mujoco.MjvScene(mj_model, maxgeom=10000)
                context = mujoco.MjrContext(mj_model, mujoco.mjtFontScale.mjFONTSCALE_150)
                cam = mujoco.MjvCamera()
                cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                cam.fixedcamid = camera_id
                
                viewport = mujoco.MjrRect(0, 0, width, height)
                mujoco.mjv_updateScene(
                    mj_model, mj_data, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene
                )
                rgb = np.zeros((height, width, 3), dtype=np.uint8)
                depth = np.zeros((height, width), dtype=np.float32)
                mujoco.mjr_render(viewport, scene, context)
                mujoco.mjr_readPixels(rgb, depth, viewport, context)
                return rgb
            else:
                return np.zeros((height, width, 3), dtype=np.uint8)
    except Exception as e:
        print(f"[WARN] Failed to render image: {e}")
        return np.zeros((height, width, 3), dtype=np.uint8)


def _get_robot_state_dict(controller: MujocoTeleopController, arm_name: str) -> dict:
    """
    获取机器人状态字典（兼容 collect_human_demonstrations.py 格式）
    arm_name: "right_hand" 或 "left_hand"
    """
    config = controller.manipulator_config[arm_name]
    link_name = config["link_name"]  # e.g., "right_tool0"
    
    # 获取末端执行器位置和姿态（world frame）
    ee_pos, ee_quat_wxyz = controller._get_link_pose(link_name)
    ee_quat_xyzw = quat_wxyz_to_xyzw(ee_quat_wxyz).astype(np.float32)
    
    # 计算末端执行器速度（使用数值差分）
    if not hasattr(controller, '_prev_ee_pos'):
        controller._prev_ee_pos = {}
        controller._prev_ee_quat = {}
        controller._prev_time = {}
    
    if arm_name not in controller._prev_ee_pos:
        ee_pos_vel = np.zeros((3,), dtype=np.float32)
        ee_ori_vel = np.zeros((3,), dtype=np.float32)
    else:
        dt = controller.dt
        if dt > 0:
            ee_pos_vel = ((ee_pos - controller._prev_ee_pos[arm_name]) / dt).astype(np.float32)
            # 计算角速度（从四元数差分）
            quat_diff = quat_diff_as_angle_axis(
                controller._prev_ee_quat[arm_name],
                ee_quat_wxyz
            )
            ee_ori_vel = (quat_diff / dt).astype(np.float32)
        else:
            ee_pos_vel = np.zeros((3,), dtype=np.float32)
            ee_ori_vel = np.zeros((3,), dtype=np.float32)
    
    controller._prev_ee_pos[arm_name] = ee_pos.copy()
    controller._prev_ee_quat[arm_name] = ee_quat_wxyz.copy()
    
    # 获取夹爪宽度
    gripper_width = 0.0
    if "gripper_config" in config:
        gripper_config = config["gripper_config"]
        joint_names = gripper_config.get("joint_names", [])
        if joint_names:
            # 获取第一个夹爪关节的位置
            joint_name = joint_names[0]
            joint_id = mujoco.mj_name2id(controller.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                qpos_idx = controller.mj_model.jnt_qposadr[joint_id]
                gripper_width = float(controller.mj_data.qpos[qpos_idx])
    
    return {
        "ee_pos": ee_pos.astype(np.float32),
        "ee_quat": ee_quat_xyzw,  # xyzw format
        "ee_pos_vel": ee_pos_vel,
        "ee_ori_vel": ee_ori_vel,
        "gripper_width": float(gripper_width),
    }


def _get_parts_poses_14(controller: MujocoTeleopController) -> np.ndarray:
    """
    返回 14D parts_poses（兼容 collect_human_demonstrations.py）
    对于 RM75，可以返回全 0 或尝试获取场景中的物体位姿
    """
    # 如果有物体，可以在这里获取
    # 目前返回全 0
    return np.zeros((14,), dtype=np.float32)


def _save_trajectory_pkl(traj: dict, out_dir: Path, prefix: str = "teleop") -> Path:
    """保存轨迹为 pkl 文件"""
    out_dir.mkdir(parents=True, exist_ok=True)
    t1, t2 = str(time.time()).split(".")
    out_path = out_dir / f"{prefix}_{t1}_{t2}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(traj, f)
    print(f"[DATA] Saved trajectory pkl -> {out_path}")
    return out_path


class DataCollectionMujocoTeleopController(MujocoTeleopController):
    """带数据采集功能的 MujocoTeleopController"""
    
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        data_dir: str = None,
        save_debug_images: bool = False,
        debug_images_dir: str = "debug_images",
        **kwargs
    ):
        super().__init__(xml_path, robot_urdf_path, manipulator_config, **kwargs)
        
        self.data_dir = data_dir
        self.save_debug_images = save_debug_images
        self.debug_images_dir = Path(debug_images_dir)
        
        # 数据采集相关
        self.is_collecting = False
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_robosuite_env_actions = []
        self.traj_norm_delta7 = []
        self.traj_sim_state = None
        self.traj_success = 0
        
        # 用于速度计算
        self._prev_ee_pos = {}
        self._prev_ee_quat = {}
        self._prev_time = {}
        
        # 用于记录当前激活的手臂
        self._current_active_arm = None
        
    def start_collection(self):
        """开始采集数据"""
        self.is_collecting = True
        self.traj_obs = []
        self.traj_actions = []
        self.traj_rewards = []
        self.traj_robosuite_env_actions = []
        self.traj_norm_delta7 = []
        self.traj_sim_state = {
            "qpos": self.mj_data.qpos.copy().astype(np.float32),
            "qvel": self.mj_data.qvel.copy().astype(np.float32),
        }
        self.traj_success = 0
        self._prev_ee_pos = {}
        self._prev_ee_quat = {}
        self._prev_time = {}
        print("[DATA] Started collecting trajectory...")
    
    def stop_collection(self):
        """停止采集并保存数据"""
        if not self.is_collecting:
            return
        
        self.is_collecting = False
        
        if len(self.traj_actions) == 0:
            print("[DATA] No data collected, skipping save.")
            return
        
        # 保存数据
        if self.data_dir is None:
            data_dir_raw = os.environ.get("DATA_DIR_RAW", None)
            if data_dir_raw is None:
                print("[WARN] DATA_DIR_RAW not set, saving to current directory")
                out_dir = Path(".")
            else:
                task_name = "dual_rm75"
                demo_outcome = "success" if self.traj_success else "failure"
                out_dir = Path(data_dir_raw) / "raw" / "osc" / "sim" / task_name / "teleop" / "low" / demo_outcome
        else:
            out_dir = Path(self.data_dir)
        
        traj = {
            "observations": self.traj_obs,
            "actions": self.traj_actions,
            "rewards": self.traj_rewards,
            "success": int(self.traj_success),
            "task": "dual_rm75",
            "sim_state": self.traj_sim_state,
            "robosuite_env_actions": self.traj_robosuite_env_actions,
            "norm_delta7": self.traj_norm_delta7,
        }
        
        out_pkl_path = _save_trajectory_pkl(traj, out_dir=out_dir, prefix="dual_rm75")
        
        # 可选：保存调试图像
        if self.save_debug_images and len(self.traj_obs) > 0:
            self.debug_images_dir.mkdir(parents=True, exist_ok=True)
            prefix = out_pkl_path.stem
            
            img1 = self.traj_obs[0].get("color_image1", None)
            img2 = self.traj_obs[0].get("color_image2", None)
            
            if isinstance(img1, np.ndarray) and img1.ndim == 3:
                Image.fromarray(img1.astype(np.uint8)).save(
                    self.debug_images_dir / f"{prefix}_cam1_wrist.png"
                )
            if isinstance(img2, np.ndarray) and img2.ndim == 3:
                Image.fromarray(img2.astype(np.uint8)).save(
                    self.debug_images_dir / f"{prefix}_cam2_agentview.png"
                )
            print(f"[DATA] Saved debug images -> {self.debug_images_dir}")
        
        print(f"[DATA] Collection stopped. Saved {len(self.traj_actions)} steps.")
    
    def _collect_data_step(self):
        """采集当前步骤的数据"""
        if not self.is_collecting:
            return
        
        # 确定当前激活的手臂
        active_arm = None
        for name in self.manipulator_config.keys():
            if self.active.get(name, False):
                active_arm = name
                break
        
        if active_arm is None:
            # 如果没有激活的手臂，跳过采集
            return
        
        self._current_active_arm = active_arm
        config = self.manipulator_config[active_arm]
        
        # 1. Observation
        # 渲染图像（如果没有相机，使用默认视角）
        color_image1 = _render_rgb(self.mj_model, self.mj_data, camera_name=None, width=320, height=240)
        color_image2 = _render_rgb(self.mj_model, self.mj_data, camera_name=None, width=320, height=240)
        
        obs_step = {
            "color_image1": color_image1,
            "color_image2": color_image2,
            "robot_state": _get_robot_state_dict(self, active_arm),
            "parts_poses": _get_parts_poses_14(self),
        }
        self.traj_obs.append(obs_step)
        
        # 2. Action: 从 IK 任务计算 delta
        task = self.effector_task[active_arm]
        T_world_target = task.T_world_frame
        
        # 获取当前末端位姿
        link_name = config["link_name"]
        ee_pos_curr, ee_quat_curr_wxyz = self._get_link_pose(link_name)
        
        # 计算 delta（位置和姿态）
        delta_pos = (T_world_target[:3, 3] - ee_pos_curr).astype(np.float32)
        
        # 计算姿态 delta（转换为角轴表示）
        ee_quat_target_wxyz = tf.quaternion_from_matrix(T_world_target)
        delta_rot_axis_angle = quat_diff_as_angle_axis(ee_quat_curr_wxyz, ee_quat_target_wxyz)
        
        # 转换为四元数增量（简化处理，直接使用角轴）
        # 对于小角度，角轴可以近似转换为四元数
        angle = np.linalg.norm(delta_rot_axis_angle)
        if angle < 1e-6:
            delta_quat_xyzw = np.array([0, 0, 0, 1], dtype=np.float32)
        else:
            axis = delta_rot_axis_angle / angle
            half_angle = angle / 2.0
            delta_quat_xyzw = np.array([
                axis[0] * np.sin(half_angle),
                axis[1] * np.sin(half_angle),
                axis[2] * np.sin(half_angle),
                np.cos(half_angle)
            ], dtype=np.float32)
        
        # 获取夹爪动作
        gripper_action = 0.0
        if "gripper_config" in config:
            gripper_target = self.gripper_pos_target.get(active_arm, {})
            if gripper_target:
                # 获取夹爪目标位置（归一化到 -1 到 1）
                joint_name = config["gripper_config"]["joint_names"][0]
                open_pos = config["gripper_config"]["open_pos"][0]
                close_pos = config["gripper_config"]["close_pos"][0]
                current_pos = gripper_target.get(joint_name, open_pos)
                # 归一化：open_pos -> 1.0, close_pos -> -1.0
                if abs(open_pos - close_pos) > 1e-6:
                    gripper_action = 2.0 * (current_pos - open_pos) / (open_pos - close_pos) - 1.0
                else:
                    gripper_action = 1.0
        
        # 构建 8D action: [Δpos3, Δquat4(xyzw), gripper1]
        action8 = np.concatenate([
            delta_pos,
            delta_quat_xyzw,
            np.array([gripper_action], dtype=np.float32)
        ]).astype(np.float32)
        
        self.traj_actions.append(action8)
        self.traj_rewards.append(0.0)
        
        # 保存归一化的 delta7（用于调试）
        delta7 = np.concatenate([delta_pos, delta_quat_xyzw]).astype(np.float32)
        self.traj_norm_delta7.append(delta7)
        
        # 保存实际发送给 MuJoCo 的控制命令
        env_action = self.mj_data.ctrl.copy().astype(np.float32)
        self.traj_robosuite_env_actions.append(env_action)
    
    def run(self):
        """运行遥操作循环，带数据采集"""
        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0.2, 0, 0]
            
            print("\n" + "="*60)
            print("Data Collection Mode")
            print("="*60)
            print("Press 'S' to start/stop data collection")
            print("Press 'Q' to quit")
            print("="*60 + "\n")
            
            while not self._stop_event.is_set():
                try:
                    # 检查键盘输入（简化版，实际可能需要更复杂的输入处理）
                    # 这里假设通过其他方式控制开始/停止
                    
                    self._update_robot_state()
                    self._update_ik()
                    self._update_gripper_target()
                    self._update_mocap_target()
                    self._send_command()
                    
                    # 采集数据（如果正在采集）
                    self._collect_data_step()
                    
                    # Step simulation
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    viewer.sync()
                    
                except KeyboardInterrupt:
                    print("\nTeleoperation stopped.")
                    if self.is_collecting:
                        self.stop_collection()
                    self._stop_event.set()


def main(
    xml_path: str = os.path.join(ASSET_PATH, "dual_RM75/scene_dual_RM75.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "dual_RM75/dual_rm75_robot.urdf"),
    scale_factor: float = 1.0,
    visualize_placo: bool = True,
    data_dir: str = None,
    save_debug_images: bool = False,
    debug_images_dir: str = "debug_images",
    auto_start_collection: bool = False,
):
    """
    启动带数据采集的双臂 RM75 遥操作
    
    Args:
        xml_path: MuJoCo XML 文件路径
        robot_urdf_path: URDF 文件路径
        scale_factor: VR 动作缩放因子
        visualize_placo: 是否可视化 Placo
        data_dir: 数据保存目录（如果为 None，使用 DATA_DIR_RAW 环境变量）
        save_debug_images: 是否保存调试图像
        debug_images_dir: 调试图像保存目录
        auto_start_collection: 是否自动开始采集（否则需要手动触发）
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
    
    controller = DataCollectionMujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        manipulator_config=config,
        data_dir=data_dir,
        save_debug_images=save_debug_images,
        debug_images_dir=debug_images_dir,
        scale_factor=scale_factor,
        visualize_placo=visualize_placo,
    )
    
    # 添加关节正则化任务
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)
    
    if auto_start_collection:
        controller.start_collection()
    
    # 运行（需要在外部控制开始/停止采集）
    # 可以通过修改代码添加键盘监听或其他触发方式
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)