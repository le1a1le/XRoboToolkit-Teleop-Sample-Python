import placo
import numpy as np
import time
import os
import meshcat.transformations as tf
from placo_utils.visualization import robot_viz
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def get_link_pose(robot, link_name):
    """获取 link 在世界坐标系下的位姿 (位置 + 四元数)"""
    try:
        T = robot.get_T_world_frame(link_name)
        # 提取位置
        pos = T[:3, 3].copy()
        
        # 提取四元数，转换为 (w, x, y, z) 格式
        quat_xyzw = tf.quaternion_from_matrix(T)  # 返回 [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        
        return pos, quat_wxyz
    except Exception as e:
        return None, None


def format_pose(pos, quat):
    """格式化位姿输出"""
    if pos is None or quat is None:
        return "N/A"
    return f"Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] | Quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"


def get_joint_angles(robot, joint_names, q_offset=0):
    """获取指定关节的关节角"""
    joint_angles = {}
    q = robot.state.q
    
    # 将 robot.joint_names() 转换为 Python list
    all_joint_names = list(robot.joint_names())
    
    for joint_name in joint_names:
        try:
            # 查找关节在 joint_names 列表中的索引
            if joint_name in all_joint_names:
                joint_idx = all_joint_names.index(joint_name)
                # 考虑 q_offset（如果是 floating base，前7个是基座）
                actual_idx = q_offset + joint_idx
                if actual_idx < len(q):
                    joint_angles[joint_name] = q[actual_idx]
        except Exception:
            pass
    
    return joint_angles

def main():
    # 加载 URDF 模型
    urdf_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.urdf")
    
    if not os.path.exists(urdf_path):
        print(f"错误: 未找到 URDF 文件: {urdf_path}")
        return
    
    robot = placo.RobotWrapper(urdf_path)
    
    # 检查是否有 floating base
    has_floating_base = robot.model.joints[1].shortname() == "JointModelFreeFlyer"
    q_offset = 7 if has_floating_base else 0
    
    # 初始化关节角
    nq = robot.model.nq
    if nq >= 7 and has_floating_base:
        # 设置固定基座的位姿和旋转（identity）
        robot.state.q[:3] = np.array([0, 0, 0])  # 位置
        robot.state.q[3:7] = np.array([0, 0, 0, 1])  # 四元数 (w, x, y, z)
    
    robot.update_kinematics()
    
    print("=" * 80)
    print("Placo URDF Tool0 Pose Monitor")
    print("=" * 80)
    print(f"URDF 文件: {urdf_path}")
    print(f"总自由度 (nq): {nq}")
    print(f"是否有 floating base: {has_floating_base}")
    print(f"关节角偏移量: {q_offset}")
    print("=" * 80)
    print("在 Meshcat 浏览器中可视化机器人模型")
    print("提示: 可以修改代码中的 robot.state.q 来改变关节角")
    print("      按 Ctrl+C 退出")
    print("=" * 80)
    print()
    
    # 创建可视化器
    viz = robot_viz(robot)
    meshcat_url = viz.viewer.url()
    print(f"Meshcat 可视化器已启动")
    print(f"请在浏览器中打开: {meshcat_url}")
    print()
    
    # 检查 tool0 links 是否存在
    try:
        left_pos, left_quat = get_link_pose(robot, "left_tool0")
        if left_pos is None:
            print("警告: 未找到 left_tool0 link")
    except Exception:
        print("警告: 无法获取 left_tool0 位姿")
    
    try:
        right_pos, right_quat = get_link_pose(robot, "right_tool0")
        if right_pos is None:
            print("警告: 未找到 right_tool0 link")
    except Exception:
        print("警告: 无法获取 right_tool0 位姿")
    
    # 定义左右臂关节名称
    left_arm_joint_names = [
        "left_shoulder_pan_joint",
        "left_shoulder_lift_joint",
        "left_elbow_joint",
        "left_wrist_1_joint",
        "left_wrist_2_joint",
        "left_wrist_3_joint",
    ]
    
    right_arm_joint_names = [
        "right_shoulder_pan_joint",
        "right_shoulder_lift_joint",
        "right_elbow_joint",
        "right_wrist_1_joint",
        "right_wrist_2_joint",
        "right_wrist_3_joint",
    ]
    
    last_print_time = time.time()
    print_interval = 1.0  # 每秒打印一次
    
    print("\n开始监控...\n")
    
    try:
        while True:
            step_start = time.time()
            
            # 更新运动学
            robot.update_kinematics()
            
            # 更新可视化
            viz.display(robot.state.q)
            
            # 实时打印 tool0 位姿和关节角（限制打印频率）
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                # 清屏并打印最新信息
                print("\033[2J\033[H", end="")  # 清屏并移动光标到顶部
                print("=" * 80)
                print("实时 Tool0 位姿和关节角 (世界坐标系)")
                print("=" * 80)
                
                # 右臂 tool0
                right_pos, right_quat = get_link_pose(robot, "right_tool0")
                print(f"\n右臂 (right_tool0):")
                print(f"  {format_pose(right_pos, right_quat)}")
                
                # 左臂 tool0
                left_pos, left_quat = get_link_pose(robot, "left_tool0")
                print(f"\n左臂 (left_tool0):")
                print(f"  {format_pose(left_pos, left_quat)}")
                
                print(f"\n当前关节角:")
                
                # 显示右臂关节
                print("  右臂关节:")
                # 将 robot.joint_names() 转换为 Python list
                all_joint_names = list(robot.joint_names())
                for joint_name in right_arm_joint_names:
                    if joint_name in all_joint_names:
                        joint_idx = all_joint_names.index(joint_name)
                        actual_idx = q_offset + joint_idx
                        if actual_idx < len(robot.state.q):
                            angle = robot.state.q[actual_idx]
                            print(f"    {joint_name:30s}: {angle:8.4f} rad ({np.degrees(angle):8.2f} deg)")
                
                # 显示左臂关节
                print("  左臂关节:")
                for joint_name in left_arm_joint_names:
                    if joint_name in all_joint_names:
                        joint_idx = all_joint_names.index(joint_name)
                        actual_idx = q_offset + joint_idx
                        if actual_idx < len(robot.state.q):
                            angle = robot.state.q[actual_idx]
                            print(f"    {joint_name:30s}: {angle:8.4f} rad ({np.degrees(angle):8.2f} deg)")

                print("\n" + "=" * 80)
                print(f"提示: 修改代码中的 robot.state.q 来改变关节角")
                print(f"      Meshcat URL: {meshcat_url}")
                print("=" * 80)
                
                last_print_time = current_time
            
            # 控制循环频率
            time.sleep(max(0, 0.01 - (time.time() - step_start)))
            
    except KeyboardInterrupt:
        print("\n\n监控已停止")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()