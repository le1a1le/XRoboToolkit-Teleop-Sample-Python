import numpy as np
import meshcat.transformations as tf
def quaternion_xyzw_to_rpy(quat_xyzw: np.ndarray) -> np.ndarray:
    """将四元数(x, y, z, w)转换为RPY欧拉角(Roll, Pitch, Yaw)。
    
    Args:
        quat_xyzw: 四元数，格式为 [x, y, z, w]
    
    Returns:
        RPY欧拉角，格式为 [roll, pitch, yaw] (弧度)
    """
    q = np.array(quat_xyzw, dtype=np.float64)
    
    # 转换为 (w, x, y, z) 格式，因为 meshcat.transformations 使用此格式
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]])
    
    # 将四元数转换为旋转矩阵
    rot_matrix = tf.quaternion_matrix(quat_wxyz)[:3, :3]
    
    # 从旋转矩阵提取欧拉角 (ZYX顺序，即RPY)
    rpy = tf.euler_from_matrix(rot_matrix, 'rzyx')
    
    return np.array([rpy[0], rpy[1], rpy[2]])  # [roll, pitch, yaw]


def rpy_to_quaternion_xyzw(rpy: np.ndarray) -> np.ndarray:
    """将RPY欧拉角(Roll, Pitch, Yaw)转换为四元数(x, y, z, w)。
    
    Args:
        rpy: RPY欧拉角，格式为 [roll, pitch, yaw] (弧度)
    
    Returns:
        四元数，格式为 [x, y, z, w]
    """
    rpy_array = np.array(rpy, dtype=np.float64)
    roll, pitch, yaw = rpy_array[0], rpy_array[1], rpy_array[2]
    
    # meshcat.transformations 使用 ZYX 顺序（即RPY）
    # quaternion_from_euler 参数顺序是 (yaw, pitch, roll) 对应 (z, y, x)
    quat_wxyz = tf.quaternion_from_euler(yaw, pitch, roll)
    
    # 转换为 (x, y, z, w) 格式
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    
    return quat_xyzw




if __name__ == "__main__":
    quat_xyzw = np.array([0.65328147  ,0.65328149 ,-0.27059805 , 0.27059805])
    print(quaternion_xyzw_to_rpy(quat_xyzw))
    # rpy = np.array([-1.5707963 ,2.35619448975, 0.0])
    # print(rpy_to_quaternion_xyzw(rpy))