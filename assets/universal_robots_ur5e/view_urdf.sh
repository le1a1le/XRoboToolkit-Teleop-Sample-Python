#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# 检查 ROS2 是否已 source
if [ -z "$ROS_DISTRO" ]; then
    echo "错误: 请先 source ROS2 环境"
    echo "例如: source /opt/ros/humble/setup.bash"
    exit 1
fi

# 启动 launch 文件
ros2 launch "$SCRIPT_DIR/view_urdf.launch.py"