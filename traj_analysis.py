import pickle
import numpy as np

# 加载轨迹文件
with open('cnn_serial_trajectory.pkl', 'rb') as f:
    serial_traj = dict(pickle.load(f))

with open('cnn_parallel_trajectory.pkl', 'rb') as f:
    parallel_traj = dict(pickle.load(f))

# 计算每一步的L2差异
diffs = {}
for step in serial_traj.keys():
    if step in parallel_traj:
        total_diff = 0
        total_params = 0
        for name in serial_traj[step].keys():
            s_param = serial_traj[step][name]
            p_param = parallel_traj[step][name]
            diff = np.linalg.norm(s_param - p_param) ** 2
            total_diff += diff
            total_params += s_param.size
        # 计算平均差异（论文中的d^t）
        diffs[step] = total_diff / total_params

# 打印结果
print("Step\tDifference (d^t)")
for step, diff in sorted(diffs.items()):
    print(f"{step}\t{diff:.6e}")