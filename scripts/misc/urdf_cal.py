import pybullet as p
import numpy as np

class UR5eIKPybullet:
    def __init__(self, urdf_path, base_position, base_orientation, is_left=False):
        """
        使用pybullet计算IK
        
        参数:
            urdf_path: URDF文件路径
            base_position: 基座位置 [x, y, z]
            base_orientation: 基座姿态（四元数）[x, y, z, w]
            is_left: 是否为左臂
        """
        self.is_left = is_left
        self.physics_client = p.connect(p.DIRECT)
        
        # 加载机器人
        self.robot_id = p.loadURDF(
            urdf_path,
            base_position,
            p.getQuaternionFromEuler(base_orientation),
            useFixedBase=True
        )
        
        # 获取关节信息
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_indices = []
        self.joint_names = []
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if 'shoulder' in joint_name or 'elbow' in joint_name or 'wrist' in joint_name:
                self.joint_indices.append(i)
                self.joint_names.append(joint_name)
        
        # 末端执行器链接索引
        self.end_effector_index = self.find_link_index('right_tool0' if not is_left else 'left_tool0')
    
    def find_link_index(self, link_name):
        """查找链接索引"""
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            if joint_info[12].decode('utf-8') == link_name:
                return i
        return -1
    
    def inverse_kinematics(self, target_position, target_orientation, max_iterations=100):
        """
        使用pybullet计算逆向运动学
        
        参数:
            target_position: 目标位置 [x, y, z] (世界坐标系)
            target_orientation: 目标姿态（四元数）[x, y, z, w]
            max_iterations: 最大迭代次数
            
        返回:
            joints: 关节角度列表 (rad)
        """
        # 使用pybullet的IK求解器
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_index,
            target_position,
            target_orientation,
            maxNumIterations=max_iterations,
            residualThreshold=1e-5
        )
        
        # 提取相关关节
        joints = []
        for idx in self.joint_indices:
            joints.append(joint_poses[idx])
        
        return joints[:6]  # 只返回前6个关节（UR5e的6个自由度）
    
    def forward_kinematics(self, joints):
        """
        使用pybullet计算正向运动学
        
        参数:
            joints: 关节角度列表 (rad)
            
        返回:
            position: 末端位置 [x, y, z]
            orientation: 末端姿态（四元数）[x, y, z, w]
        """
        # 设置关节角度
        for i, idx in enumerate(self.joint_indices[:6]):
            p.resetJointState(self.robot_id, idx, joints[i])
        
        # 计算正向运动学
        link_state = p.getLinkState(self.robot_id, self.end_effector_index)
        position = link_state[0]
        orientation = link_state[1]
        
        return np.array(position), np.array(orientation)
    
    def close(self):
        """关闭物理引擎"""
        p.disconnect(self.physics_client)


# 使用示例
def use_pybullet_ik():
    urdf_path = "/home/zz/mujoco_RL_file/robosuite/pico_vr/XRoboToolkit-Teleop-Sample-Python/assets/universal_robots_ur5e/dual_ur5e.urdf"
    
    # 右臂
    right_arm_ik = UR5eIKPybullet(
        urdf_path,
        base_position=[0, 0, 0],
        base_orientation=[0, 0, 0],
        is_left=False
    )
    
    # 目标位姿
    target_pos = [0.5, 0.2, 0.5]
    target_orn = p.getQuaternionFromEuler([0, np.pi/2, 0])
    
    # 计算IK
    joints = right_arm_ik.inverse_kinematics(target_pos, target_orn)
    print(f"计算得到的关节角: {joints}")
    
    # 验证正向运动学
    calc_pos, calc_orn = right_arm_ik.forward_kinematics(joints)
    print(f"正向运动学验证位置: {calc_pos}")
    print(f"正向运动学验证姿态: {calc_orn}")
    
    right_arm_ik.close()

if __name__ == "__main__":
    use_pybullet_ik()