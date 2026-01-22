from typing import Any, Dict

import mujoco
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.mujoco_utils import (
    calc_mujoco_ctrl_from_qpos,
    calc_mujoco_qpos_from_placo_q,
    calc_placo_q_from_mujoco_qpos,
    set_mujoco_joint_pos_by_name,
)


class MujocoTeleopController(BaseTeleopController):
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        manipulator_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
        dt=0.01,
        mj_qpos_init=None,
    ):
        self.visualize_placo = visualize_placo
        self.xml_path = xml_path
        self.mj_qpos_init = mj_qpos_init

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.target_mocap_idx = {name: -1 for name in manipulator_config.keys()}

        super().__init__(
            robot_urdf_path,
            manipulator_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init=None,
            dt=dt,
        )

        if visualize_placo:
            self._init_placo_viz()

    def _robot_setup(self):
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]   
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.mj_qpos_init is None:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, self.mj_model.key("home").id)
        else:
            self.mj_data.qpos[:] = self.mj_qpos_init
            self.mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(self.mj_model, self.mj_qpos_init)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        for _ in range(10):
            mujoco.mj_step(self.mj_model, self.mj_data)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # setup mocap target
        for name, config in self.manipulator_config.items():
            if "vis_target" not in config:
                print(f"Warning: 'vis_target' not found in config for {name}. Skipping mocap setup.")
                continue
            vis_target = config["vis_target"]
            mocap_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis_target)
            if mocap_id == -1:
                raise ValueError(f"Mocap body '{vis_target}' not found in the model.")

            if self.mj_model.body_mocapid[mocap_id] == -1:
                raise ValueError(f"Body '{vis_target}' is not configured for mocap.")
            else:
                self.target_mocap_idx[name] = self.mj_model.body_mocapid[mocap_id]

            print(f"Mocap ID for '{vis_target}' body: {self.target_mocap_idx[name]}")

                    # 初始化 mocap 位置为 XML 中定义的值
            mocap_idx = self.target_mocap_idx[name]
            # 获取 XML 中定义的初始位置和姿态
            body_pos = self.mj_model.body_pos[mocap_id].copy()
            body_quat = self.mj_model.body_quat[mocap_id].copy()
            self.mj_data.mocap_pos[mocap_idx] = body_pos
            self.mj_data.mocap_quat[mocap_idx] = body_quat
            print(f"Initialized {vis_target} mocap position: {body_pos}, quat: {body_quat}")

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver."""
        # 先调用父类方法设置 placo_robot
        super()._placo_setup()
                # 在 Placo 设置完成后，从 MuJoCo 同步初始关节值
        mj_qpos = self.mj_data.qpos.copy()
        self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
            self.mj_model,
            self.placo_robot,
            mj_qpos,
            floating_base=self.floating_base,
        )
        self.placo_robot.update_kinematics()
        
        # 重新同步任务目标到当前末端执行器位置（使用正确的关节值计算）
        self.sync_end_effector_poses_to_placo_tasks()
        
        print("Placo initialized with MuJoCo home position and tasks synced.")
        print("=" * 80)
        print("Placo Joint Information:")
        print("=" * 80)
        pin_joint_names = [name for name in self.placo_robot.model.names if name not in ["root_joint", "universe"]]
        print(f"Total actuated joints: {len(pin_joint_names)}")
        for i, joint_name in enumerate(pin_joint_names):
            print(f"  [{i:2d}] {joint_name}")
        print("=" * 80)
        
        # 在 placo 设置完成后，将 MuJoCo 的初始 qpos 同步到 placo_robot
        if self.mj_model is not None and self.mj_data is not None:
            mj_qpos = self.mj_data.qpos.copy()
            placo_q_before = self.placo_robot.state.q.copy()
            self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
                self.mj_model,
                self.placo_robot,
                mj_qpos,
                floating_base=self.floating_base,
            )
            self.placo_robot.update_kinematics()
            placo_q_after = self.placo_robot.state.q.copy()
            
            print("\n" + "=" * 80)
            print("Initial State Synchronization (MuJoCo -> Placo):")
            print("=" * 80)
            pin_joint_names = [name for name in self.placo_robot.model.names if name not in ["root_joint", "universe"]]
            pin_q_offset = 7 if not self.floating_base and self.placo_robot.model.joints[1].shortname() == "JointModelFreeFlyer" else 0
            
            for i, joint_name in enumerate(pin_joint_names):
                mj_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if mj_joint_id != -1:
                    mj_qpos_addr = self.mj_model.jnt_qposadr[mj_joint_id]
                    mj_value = mj_qpos[mj_qpos_addr] if mj_qpos_addr < len(mj_qpos) else 0.0
                else:
                    mj_value = None
                    mj_qpos_addr = -1
                
                placo_idx = i + pin_q_offset
                placo_value_before = placo_q_before[placo_idx] if placo_idx < len(placo_q_before) else 0.0
                placo_value_after = placo_q_after[placo_idx] if placo_idx < len(placo_q_after) else 0.0
                
                status = "✓" if mj_joint_id != -1 else "✗"
                mj_value_str = f"{mj_value:8.4f}" if mj_value is not None else "      N/A"
                print(f"  {status} [{i:2d}] {joint_name:40s} | "
                    f"MuJoCo[{mj_qpos_addr:2d}]={mj_value_str} | "
                    f"Placo[{placo_idx:2d}] {placo_value_before:8.4f} -> {placo_value_after:8.4f}")
            print("=" * 80)
    def _send_command(self):
        placo_q = self.placo_robot.state.q.copy()
        qpos_desired = calc_mujoco_qpos_from_placo_q(
            self.mj_model,
            self.placo_robot,
            self.placo_robot.state.q,
            floating_base=self.floating_base,
        )
        # 只在第一次循环时打印
        if not hasattr(self, '_first_send_printed'):
            self._first_send_printed = True
            print("\n" + "=" * 80)
            print("First _send_command() - Placo -> MuJoCo:")
            print("=" * 80)
            pin_joint_names = [name for name in self.placo_robot.model.names if name not in ["root_joint", "universe"]]
            pin_q_offset = 7 if (not self.floating_base and self.placo_robot.model.joints[1].shortname() == "JointModelFreeFlyer") else 0
            
            for i, joint_name in enumerate(pin_joint_names):
                mj_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if mj_joint_id != -1:
                    mj_qpos_addr = self.mj_model.jnt_qposadr[mj_joint_id]
                    mj_value_before = self.mj_data.qpos[mj_qpos_addr] if mj_qpos_addr < len(self.mj_data.qpos) else 0.0
                    mj_value_after = qpos_desired[mj_qpos_addr] if mj_qpos_addr < len(qpos_desired) else 0.0
                else:
                    mj_value_before = None
                    mj_value_after = None
                    mj_qpos_addr = -1
                
                placo_idx = i + pin_q_offset
                placo_value = placo_q[placo_idx] if placo_idx < len(placo_q) else 0.0
                
                status = "✓" if mj_joint_id != -1 else "✗"
                if mj_value_before is not None and mj_value_after is not None:
                    diff = abs(mj_value_after - mj_value_before)
                    change_marker = "***" if diff > 0.01 else ""
                    print(f"  {status} [{i:2d}] {joint_name:40s} | "
                          f"Placo[{placo_idx:2d}]={placo_value:8.4f} | "
                          f"MuJoCo[{mj_qpos_addr:2d}] {mj_value_before:8.4f} -> {mj_value_after:8.4f} {change_marker}")
                else:
                    print(f"  {status} [{i:2d}] {joint_name:40s} | "
                          f"Placo[{placo_idx:2d}]={placo_value:8.4f} | "
                          f"MuJoCo: NOT FOUND")
            print("=" * 80)
        for gripper_name, gripper_target in self.gripper_pos_target.items():
            for joint_name, joint_pos in gripper_target.items():
                success = set_mujoco_joint_pos_by_name(
                    self.mj_model,
                    qpos_desired,
                    joint_name,
                    joint_pos,
                )
                if not success:
                    raise ValueError(f"Joint '{gripper_name}' not found in MuJoCo model.")

        self.mj_data.ctrl = calc_mujoco_ctrl_from_qpos(self.mj_model, qpos_desired)

        if self.visualize_placo:
            self._update_placo_viz()

    def _update_robot_state(self):
        # 获取最新的关节角
        mj_qpos = self.mj_data.qpos.copy()
        placo_q_before = self.placo_robot.state.q.copy()
        self.placo_robot.state.q = calc_placo_q_from_mujoco_qpos(
            self.mj_model,
            self.placo_robot,
            mj_qpos,
            floating_base=self.floating_base,
        )
        self.placo_robot.update_kinematics()
        placo_q_after = self.placo_robot.state.q.copy()
        
        # 只在第一次循环或有变化时打印
        if not hasattr(self, '_first_update_printed'):
            self._first_update_printed = True
            print("\n" + "=" * 80)
            print("First _update_robot_state() - MuJoCo -> Placo:")
            print("=" * 80)
            pin_joint_names = [name for name in self.placo_robot.model.names if name not in ["root_joint", "universe"]]
            pin_q_offset = 7 if (not self.floating_base and self.placo_robot.model.joints[1].shortname() == "JointModelFreeFlyer") else 0
            
            for i, joint_name in enumerate(pin_joint_names):
                mj_joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if mj_joint_id != -1:
                    mj_qpos_addr = self.mj_model.jnt_qposadr[mj_joint_id]
                    mj_value = mj_qpos[mj_qpos_addr] if mj_qpos_addr < len(mj_qpos) else 0.0
                else:
                    mj_value = None
                    mj_qpos_addr = -1
                
                placo_idx = i + pin_q_offset
                placo_value_before = placo_q_before[placo_idx] if placo_idx < len(placo_q_before) else 0.0
                placo_value_after = placo_q_after[placo_idx] if placo_idx < len(placo_q_after) else 0.0
                
                status = "✓" if mj_joint_id != -1 else "✗"
                diff = abs(placo_value_after - placo_value_before)
                change_marker = "***" if diff > 0.01 else ""
                mj_vale = f"{mj_value:8.4f}" if mj_value is not None else 'N/A'
                print(f"  {status} [{i:2d}] {joint_name:40s} | "
                      f"MuJoCo[{mj_qpos_addr:2d}]={mj_vale} | "
                      f"Placo[{placo_idx:2d}] {placo_value_before:8.4f} -> {placo_value_after:8.4f} {change_marker}")
            print("=" * 80)

    def _update_mocap_target(self):
        for name, task in self.effector_task.items():
            T_world_target = task.T_world_frame
            mocap_idx = self.target_mocap_idx.get(name)
            if mocap_idx is not None and mocap_idx != -1:
                self.mj_data.mocap_pos[mocap_idx] = T_world_target[:3, 3]
                self.mj_data.mocap_quat[mocap_idx] = tf.quaternion_from_matrix(T_world_target)

    def _get_link_pose(self, ee_name):
        """Get the end effector position and orientation."""
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name)
        if ee_id == -1:
            raise ValueError(f"End effector body '{ee_name}' not found in the model.")

        ee_xyz = self.mj_data.xpos[ee_id].copy()
        ee_quat = self.mj_data.xquat[ee_id].copy()

        return ee_xyz, ee_quat

    def run(self):
        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Set up viewer camera
            viewer.cam.azimuth = 180
            viewer.cam.elevation = -65
            viewer.cam.distance = 1.3
            viewer.cam.lookat = [-0.65, 0, 1.45]

            while not self._stop_event.is_set():
                try:
                    self._update_robot_state()
                    self._update_ik()
                    self._update_gripper_target()
                    self._update_mocap_target()
                    self._send_command()

                    # Step simulation and update viewer
                    mujoco.mj_step(self.mj_model, self.mj_data)
                    viewer.sync()
                except KeyboardInterrupt: 
                    print("\nTeleoperation stopped.")   
                    self._stop_event.set()
