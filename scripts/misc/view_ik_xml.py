import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

def get_body_pose(model, data, body_name):
    """获取 body 在世界坐标系下的位姿 (位置 + 四元数)"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None, None
    
    # 获取位置
    pos = data.xpos[body_id].copy()
    
    # 获取四元数 (w, x, y, z)
    quat = data.xquat[body_id].copy()  # MuJoCo 使用 (w, x, y, z) 格式
    
    return pos, quat

def format_pose(pos, quat):
    """格式化位姿输出"""
    if pos is None or quat is None:
        return "N/A"
    return f"Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] | Quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"

def main():
    # 加载模型
    xml_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/dual_ur5e.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 重置到 home keyframe
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        home_key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(model, data, home_key_id)
    
    mujoco.mj_forward(model, data)
    
    print("=" * 80)
    print("MuJoCo RM75 Tool0 Pose Monitor")
    print("=" * 80)
    print("使用 MuJoCo viewer 交互式控制关节:")
    print("  - 鼠标左键拖拽: 旋转视角")
    print("  - 鼠标右键拖拽: 平移视角")
    print("  - 鼠标滚轮: 缩放")
    print("  - 点击关节后按 Ctrl/Cmd + 鼠标拖拽: 控制关节")
    print("  - 按 ESC 退出")
    print("=" * 80)
    print()
    
    # 检查 tool0 body 是否存在
    right_tool0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_tool0")
    left_tool0_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_tool0")
    
    if right_tool0_id < 0:
        print("警告: 未找到 right_tool0 body")
    if left_tool0_id < 0:
       print("警告: 未找到 left_tool0 body")
     
    last_print_time = time.time()
    print_interval = 1  # 每 0.1 秒打印一次，避免刷屏
     
    # 启动交互式 viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 同步 viewer 和 data
            step_start = time.time()
            
            # 执行一步仿真（如果 viewer 没有自动步进）
            mujoco.mj_step(model, data)
            
            # 实时打印 tool0 位姿（限制打印频率）
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                # 清屏并打印最新位姿
                print("\033[2J\033[H", end="")  # 清屏并移动光标到顶部 
                print("=" * 80)
                print("实时 Tool0 位姿 (世界坐标系)")
                print("=" * 80)
                
                # 右臂 tool0
                right_pos, right_quat = get_body_pose(model, data, "right_tool0")
                print(f"\n右臂 (right_tool0):")
                print(f"  {format_pose(right_pos, right_quat)}")
                
                # 左臂 tool0
                left_pos, left_quat = get_body_pose(model, data, "left_tool0")
                print(f"\n左臂 (left_tool0):")
                print(f"  {format_pose(left_pos, left_quat)}")
                
                # 显示当前关节角（可选）
                print(f"\n当前关节角 (前7个为右臂, 后7个为左臂):")
                joint_names = []
                for i in range(model.njnt):
                    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    if joint_name:
                        joint_names.append(joint_name)
                
                # 显示右臂关节
                print("  右臂关节:")
                for i, name in enumerate(joint_names[:7]):
                    if i < len(data.qpos):
                        print(f"    {name}: {data.qpos[i]:.4f} rad")
                
                # 显示左臂关节
                print("  左臂关节:")
                left_start_idx = 7  # 假设右臂7个关节后是左臂
                for i, name in enumerate(joint_names[7:14]):
                    idx = left_start_idx + i
                    if idx < len(data.qpos):
                        print(f"    {name}: {data.qpos[idx]:.4f} rad")
                
                print("\n" + "=" * 80)
                print("提示: 在 viewer 中点击关节后按 Ctrl/Cmd + 拖拽鼠标控制关节")
                print("=" * 80)
                
                last_print_time = current_time
            
            # 同步 viewer（确保 viewer 显示最新状态）
            viewer.sync()
            
            # 控制循环频率
            time.sleep(max(0, 0.01 - (time.time() - step_start)))

if __name__ == "__main__":
    main()