from ast import main
import os
import numpy as np
import placo
import meshcat.transformations as tf

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

def solve_rm75_ik_right(T_world_tool0: np.ndarray):
    """
    输入:
        T_world_tool0: 4x4 齐次变换矩阵, 表示 right_tool0 在 world 坐标系下的位姿
    输出:
        q_solution: np.ndarray, 包含整个双臂机器人所有关节角 (rad)
                    其中右臂 7 个关节在对应的 Placo 关节索引里
    """
    # 1. 加载 URDF
    urdf_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf")
    robot = placo.RobotWrapper(urdf_path)

    # 2. 创建 KinematicsSolver
    solver = placo.KinematicsSolver(robot)
    solver.mask_fbase(True)       # 固定基座
    # 移除这行：solver.add_kinetic_energy_regularization_task(1e-6)

    # 3. 设置初始关节角（可以用全 0 或者你喜欢的初始姿态）
    q_init = np.zeros(robot.model.nq)
    robot.state.q = q_init
    robot.update_kinematics()

    # 4. 定义右臂末端的任务 (link_name 要和 URDF 里保持一致，如 right_tool0)
    ee_link_name = "right_tool0"   # 你在 URDF 里刚加的 link
    task = solver.add_frame_task(ee_link_name, T_world_tool0)
    task.configure("right_ee_task", "soft", 1.0)

    # 5. 可选: 添加一个轻微的关节正则化，使解更平滑
    joints_task = solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in robot.joint_names()})
    joints_task.configure("joints_reg", "soft", 1e-4)

    # 6. 迭代求解 IK
    for _ in range(200):
        solver.solve(True)

    # 7. 读取最终关节解
    q_solution = robot.state.q.copy()
    return q_solution

def pose_to_T(xyz, quat_wxyz):
    T = tf.quaternion_matrix(quat_wxyz)   # 4x4
    T[:3, 3] = np.array(xyz)
    return T

if __name__ == "__main__":
    xyz = [-0.4170 ,-0.4914 ,1.3000]
    quat_wxyz = [-0.0090 ,-0.8192 ,-0.5649 ,0.0986]
    T = pose_to_T(xyz, quat_wxyz)
    print(T)
    
    # 先加载 robot 来获取关节名称
    urdf_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf")
    robot = placo.RobotWrapper(urdf_path)
    
    q_solution = solve_rm75_ik_right(T)
    
    print("\n" + "="*80)
    print(f"总共有 {len(q_solution)} 个关节 (robot.model.nq = {robot.model.nq})")
    print("="*80)
    
    # 打印所有关节名称和对应的值
    joint_names = robot.joint_names()
    print(f"\n关节名称列表 (共 {len(joint_names)} 个):")
    for i, name in enumerate(joint_names):
        print(f"  {i}: {name}")
    
    print(f"\n关节角值 (qpos):")
    for i, (name, value) in enumerate(zip(joint_names, q_solution)):
        print(f"  [{i:2d}] {name:30s} = {value:10.6f} rad ({np.degrees(value):8.3f} deg)")
    
    # 如果 q_solution 长度和 joint_names 不一致，说明有额外的自由度
    if len(q_solution) != len(joint_names):
        print(f"\n注意: q_solution 有 {len(q_solution)} 个元素，但只有 {len(joint_names)} 个关节名称")
        print("额外的元素可能是:")
        if len(q_solution) > len(joint_names):
            for i in range(len(joint_names), len(q_solution)):
                print(f"  [{i:2d}] (无名称) = {q_solution[i]:10.6f}")
    
    print("\n" + "="*80)
    print("关节分组:")
    print("="*80)
    
    # 尝试按左右臂分组显示
    right_joints = [name for name in joint_names if name.startswith("right_")]
    left_joints = [name for name in joint_names if name.startswith("left_")]
    other_joints = [name for name in joint_names if not (name.startswith("right_") or name.startswith("left_"))]
    
    print(f"\n右臂关节 ({len(right_joints)} 个):")
    for name in right_joints:
        idx = joint_names.index(name)
        print(f"  {name:30s} = {q_solution[idx]:10.6f} rad")
    
    print(f"\n左臂关节 ({len(left_joints)} 个):")
    for name in left_joints:
        idx = joint_names.index(name)
        print(f"  {name:30s} = {q_solution[idx]:10.6f} rad")
    
    if other_joints:
        print(f"\n其他关节 ({len(other_joints)} 个):")
        for name in other_joints:
            idx = joint_names.index(name)
            print(f"  {name:30s} = {q_solution[idx]:10.6f} rad")