from scipy.spatial.transform import Rotation as R
import numpy as np

def convert_PosQuat2PosRotationMatrix_batch(pos_quat_gripper, quat_order="xyzw"):
    """
    - input shape: (N, 16) = [left_xyz(3) + left_quat(4) + left_gripper(1) + right_xyz(3) + right_quat(4) + right_gripper(1)]
    - output shape: (N, 20) = [left_xyz(3) + left_rotmat(6) + left_gripper(1) + right_xyz(3) + right_rotmat(6) + right_gripper(1)]
    """
    assert quat_order == "xyzw"
    N = pos_quat_gripper.shape[0]
    output = np.zeros((N, 20), dtype=pos_quat_gripper.dtype)

    # Process left arm
    left_pos = pos_quat_gripper[:, :3]
    left_quat = pos_quat_gripper[:, 3:7]
    left_gripper = pos_quat_gripper[:, 7:8]

    left_rotation = R.from_quat(left_quat)
    left_matrix = left_rotation.as_matrix()
    output[:, 0:3] = left_pos
    output[:, 3:6] = left_matrix[:, 0, :]  # first row of rotation matrix
    output[:, 6:9] = left_matrix[:, 1, :]  # second row of rotation matrix
    output[:, 9:10] = left_gripper

    # Process right arm
    right_pos = pos_quat_gripper[:, 8:11]
    right_quat = pos_quat_gripper[:, 11:15]
    right_gripper = pos_quat_gripper[:, 15:16]

    right_rotation = R.from_quat(right_quat)
    right_matrix = right_rotation.as_matrix()
    output[:, 10:13] = right_pos
    output[:, 13:16] = right_matrix[:, 0, :]  # first row of rotation matrix
    output[:, 16:19] = right_matrix[:, 1, :]  # second row of rotation matrix
    output[:, 19:20] = right_gripper

    return output


def convert_PosQuat2PosRotationMatrix(pos, quat, gripper, quat_order):
    """
    assembling pos, rotation and gripper
    - rotation will be converted from quaternion(4dim, xyzw) to 2 rows of rotation matrix(6dim)
    """
    # quat: [x, y, z, w]
    assert quat_order == "xyzw"
    ee_out = np.ones(10, dtype=pos.dtype)
    rotation = R.from_quat(quat)  # 注意：输入顺序必须是 [x, y, z, w]
    matrix = rotation.as_matrix()
    ee_out[0:3] = pos.copy()
    ee_out[3:6] = matrix[0, :]
    ee_out[6:9] = matrix[1, :]
    ee_out[9] = gripper
    return ee_out


def convert_PosRotationMatrix2PosQuat(posRM, quat_order):
    """
    convert position/rotation matrix (2 row)/gripper to position/quaternion/gripper
    """
    # quat: [x, y, z, w]
    assert quat_order == "xyzw"
    joint_out = np.ones(8, dtype=posRM.dtype)
    pos = posRM[0:3]
    c0 = posRM[3:6]
    c1 = posRM[6:9]
    c2 = cross(c0, c1)

    rotation_matrix = np.stack((c0, c1, c2), axis=0)
    rotation = R.from_matrix(rotation_matrix)
    quat = rotation.as_quat()  # xyzw

    joint_out[0:3] = pos.copy()
    joint_out[3:7] = quat.copy()
    joint_out[7] = posRM[9]
    return joint_out


def unwrap_euler_angles(current_euler, last_euler):
    # Apply unwrap_angle to each component of the Euler angles
    unwrapped_euler = np.array([unwrap_angle(current_euler[i], last_euler[i]) for i in range(3)])
    return unwrapped_euler


def unwrap_angle(angle, prev_angle):
    # Calculate the difference
    delta = angle - prev_angle
    # Adjust the angle to avoid discontinuity
    if delta > 180:
        angle -= 360
    elif delta < -180:
        angle += 360
    return angle


def convert_PosEuler2PosRotationMatrix(pos, euler, gripper):
    """
    assembling pos, rotation and gripper
    note that - rotation will be converted from euler(3dim) to 2 row of rotation matrix(6dim)
    """
    ee_out = np.ones_like(pos, shape=[10])
    rotation = R.from_euler('xyz', euler, degrees=True)
    matrix = rotation.as_matrix()

    ee_out[0:3] = pos.copy()
    ee_out[3:6] = matrix[0, :]
    ee_out[6:9] = matrix[1, :]
    ee_out[9] = gripper
    return ee_out


def convert_PosRotationMatrix2PosEuler(posRM):
    """
    convert position/rotation matrix (2 row)/gripper to position/euler/gripper
    """
    joint_out = np.ones_like(posRM, shape=[7])
    pos = posRM[0:3]
    c0 = posRM[3:6]
    c1 = posRM[6:9]
    c2 = cross(c0, c1)

    rotation_matrix = np.stack((c0, c1, c2), axis=0)
    rotation = R.from_matrix(rotation_matrix)
    euler = rotation.as_euler('xyz', degrees=True)

    joint_out[0:3] = pos.copy()
    joint_out[3:6] = euler.copy()
    joint_out[6] = posRM[9]
    return joint_out


def cross(v1, v2):
    """
    :param v1: ...x3
    :param v2: ...x3
    :return: return v3 from cross product, which is normalized vector, ...x3
    """
    v1_normalized = v1 / np.linalg.norm(v1)
    v2_normalized = v2 / np.linalg.norm(v2)
    v3 = np.cross(v1_normalized, v2_normalized)
    v3_normalized = v3 / np.linalg.norm(v3)
    return v3_normalized


def calculate_relative_poses_aligned(poses: np.ndarray) -> np.ndarray:
    num_poses = poses.shape[0]
    if num_poses == 0:
        return np.empty((0, 7))

    relative_poses_out = np.zeros((num_poses, 7))
    relative_poses_out[0, 6] = 1.0

    for i in range(num_poses - 1):
        pose_prev = poses[i]
        pose_curr = poses[i + 1]

        t_prev, q_prev = pose_prev[:3], pose_prev[3:]
        t_curr, q_curr = pose_curr[:3], pose_curr[3:]

        rot_prev = R.from_quat(q_prev)
        rot_curr = R.from_quat(q_curr)

        rot_rel = rot_prev.inv() * rot_curr
        q_rel = rot_rel.as_quat()

        t_rel = rot_prev.inv().apply(t_curr - t_prev)

        relative_poses_out[i + 1] = np.concatenate([t_rel, q_rel])

    return relative_poses_out


def _get_relative_pose(states: np.ndarray) -> np.ndarray:
    num_states = states.shape[0]
    if num_states == 0:
        return np.empty((0, 16))

    if states.shape[1] != 16:
        raise ValueError(f"Input array must have 16 columns, but got {states.shape[1]}")

    left_ee_poses = states[:, 0:7]
    left_grippers = states[:, 7:8]
    right_ee_poses = states[:, 8:15]
    right_grippers = states[:, 15:16]

    relative_left_poses = calculate_relative_poses_aligned(left_ee_poses)
    relative_right_poses = calculate_relative_poses_aligned(right_ee_poses)

    relative_states = np.concatenate([
        relative_left_poses,
        left_grippers,
        relative_right_poses,
        right_grippers
    ], axis=1)

    return relative_states


def get_relative_xyz(states):
    if len(states) == 0:
        return np.array([], dtype=np.float32)

    states = states.astype(np.float32)

    N = states.shape[0]
    relative_states = np.zeros_like(states, dtype=np.float32)

    relative_states[0] = states[0]
    relative_states[0, 0:3] = 0
    relative_states[0, 8:11] = 0

    for i in range(1, N):
        # 左臂处理
        relative_states[i, 0:3] = states[i, 0:3] - states[i - 1, 0:3]
        relative_states[i, 3:7] = states[i, 3:7]
        relative_states[i, 7] = states[i, 7]

        # 右臂处理
        relative_states[i, 8:11] = states[i, 8:11] - states[i - 1, 8:11]
        relative_states[i, 11:15] = states[i, 11:15]
        relative_states[i, 15] = states[i, 15]

    return relative_states


def get_relative_pose(poses):
    '''
    input: (N, 16)
    output relative: (N, 20)
    '''
    N = poses.shape[0]
    delta_poses = np.zeros((N, 20))

    delta_poses[:, 9] = poses[:, 7]
    delta_poses[:, 19] = poses[:, 15]

    delta_pos_left = poses[1:, :3] - poses[:-1, :3]
    rotations_left = R.from_quat(poses[:, 3:7])
    R_rel_left = rotations_left[1:] * rotations_left[:-1].inv()
    rot_elements_left = R_rel_left.as_matrix()[:, :2, :].reshape(-1, 6)

    delta_pos_right = poses[1:, 8:11] - poses[:-1, 8:11]
    rotations_right = R.from_quat(poses[:, 11:15])
    R_rel_right = rotations_right[1:] * rotations_right[:-1].inv()
    rot_elements_right = R_rel_right.as_matrix()[:, :2, :].reshape(-1, 6)

    # 填充结果
    delta_poses[1:, :3] = delta_pos_left
    delta_poses[1:, 3:9] = rot_elements_left
    delta_poses[1:, 10:13] = delta_pos_right
    delta_poses[1:, 13:19] = rot_elements_right

    return delta_poses


def _get_relative_chunk_pose_dual_arm(poses):
    """
    input: poses (N, 20)
    output: delta_poses (N, 20)
    """
    N = poses.shape[0]
    delta_poses = np.zeros((N, 20))

    for arm_idx in range(2):
        start_idx = arm_idx * 10
        end_idx = start_idx + 10

        arm_poses = poses[:, start_idx:end_idx]

        delta_pos = arm_poses[:, :3] - arm_poses[0, :3]

        R_input = arm_poses[:, 3:9].reshape(N, 2, 3)

        row0 = R_input[:, 0, :]
        row1 = R_input[:, 1, :]
        row2 = np.cross(row0, row1)

        R_full = np.stack([row0, row1, row2], axis=1)

        # 计算相对旋转矩阵
        R_rel = R_full @ R_full[0].T

        rot_elements = R_rel[:, :2, :].reshape(N, 6)

        gripper = arm_poses[:, 9:10]

        arm_delta_poses = np.hstack([delta_pos, rot_elements, gripper])
        delta_poses[:, start_idx:end_idx] = arm_delta_poses

    return delta_poses


def get_relative_chunk_pose_dual_arm(poses):
    N = poses.shape[0]
    delta_poses = np.zeros((N, 20))

    # A small epsilon to prevent division by zero during normalization
    epsilon = 1e-10

    for arm_idx in range(2):
        start_idx = arm_idx * 10
        end_idx = start_idx + 10

        arm_poses = poses[:, start_idx:end_idx]

        delta_pos = arm_poses[:, :3] - arm_poses[0, :3]

        R_input = arm_poses[:, 3:9].reshape(N, 2, 3)

        b1_raw = R_input[:, 0, :]
        b2_raw = R_input[:, 1, :]

        b1 = b1_raw / (np.linalg.norm(b1_raw, axis=-1, keepdims=True) + epsilon)
        dot_product = np.sum(b2_raw * b1, axis=-1, keepdims=True)
        b2_orthogonal = b2_raw - dot_product * b1
        b2 = b2_orthogonal / (np.linalg.norm(b2_orthogonal, axis=-1, keepdims=True) + epsilon)
        b3 = np.cross(b1, b2)
        R_full = np.stack([b1, b2, b3], axis=1)
        R_rel = R_full @ R_full[0].T
        rot_elements = R_rel[:, :2, :].reshape(N, 6)

        gripper = arm_poses[:, 9:10]
        arm_delta_poses = np.hstack([delta_pos, rot_elements, gripper])

        delta_poses[:, start_idx:end_idx] = arm_delta_poses

    return delta_poses


# def poses_to_relative_matrices(pose_sequence):
#     num_poses = pose_sequence.shape[0]
#     delta_T_list = []

#     # 1. 获取(第一帧) 的位姿并计算其逆变换 T0_inv
#     pos0 = pose_sequence[0, :3]
#     quat0_xyzw = pose_sequence[0, 3:]

#     r0 = R.from_quat(quat0_xyzw)

#     """
#     t0 = np.eye(4)
#     t0[rotation] == xx
#     t0[position] = xx

#     t0_inv = invers(to_inv)

#     func - get_relative_to()
#     """

#     R0_T = r0.as_matrix().T
#     t0_inv = -R0_T @ pos0

#     T0_inv = np.eye(4)
#     T0_inv[:3, :3] = R0_T
#     T0_inv[:3, 3] = t0_inv

#     # 2. 遍历所有帧，计算 delta_T
#     for i in range(num_poses):

#         pos_i = pose_sequence[i, :3]
#         quat_i_xyzw = pose_sequence[i, 3:]
#         Ti = np.eye(4)
#         Ti[:3, :3] = R.from_quat(quat_i_xyzw).as_matrix()
#         Ti[:3, 3] = pos_i
#         delta_T_i = T0_inv @ Ti

#         # 提取
#         rotation_6d = delta_T_i[:2, :3].flatten()
#         translation_3d = delta_T_i[:3, 3]
#         compressed_T = np.concatenate([rotation_6d, translation_3d])
#         delta_T_list.append(compressed_T)

#     return np.array(delta_T_list)

def poses_to_relative_matrices(pose_sequence):
    positions = pose_sequence[:, :3]
    quats = pose_sequence[:, 3:]

    r_all = R.from_quat(quats)
    R_all_matrices = r_all.as_matrix()

    # 获取第0帧的逆变换
    R0_T = R_all_matrices[0].T  # R0^T
    pos0 = positions[0]
    t0_inv = -R0_T @ pos0

    # 构建 T0_inv
    T0_inv = np.eye(4)
    T0_inv[:3, :3] = R0_T
    T0_inv[:3, 3] = t0_inv

    # 3. 批量构建 Ti (N, 4, 4)
    N = pose_sequence.shape[0]
    Ti_all = np.eye(4).reshape(1, 4, 4).repeat(N, axis=0)
    Ti_all[:, :3, :3] = R_all_matrices
    Ti_all[:, :3, 3] = positions

    # 批量矩阵乘法: T0_inv @ Ti
    # (4, 4) @ (N, 4, 4)
    delta_T_all = T0_inv @ Ti_all  # Shape: (N, 4, 4)

    rotation_6d = delta_T_all[:, :2, :3].reshape(N, -1)
    translation_3d = delta_T_all[:, :3, 3]

    return np.concatenate([translation_3d, rotation_6d], axis=1)


def poses_to_relative_matrices_adjacent(pose_sequence):
    num_poses = pose_sequence.shape[0]
    if num_poses == 0:
        return np.empty((0, 9))

    delta_T_list = []

    # 单位变换
    identity_matrix = np.eye(4)
    rotation_6d_identity = identity_matrix[:2, :3].flatten()  # [1, 0, 0, 0, 1, 0]
    translation_3d_identity = identity_matrix[:3, 3]  # [0, 0, 0]
    compressed_identity = np.concatenate([rotation_6d_identity, translation_3d_identity])
    delta_T_list.append(compressed_identity)

    # 计算相邻变换
    for i in range(1, num_poses):
        # 获取前一帧 (i-1) 的位姿并计算其逆变换 T_prev_inv
        pose_prev = pose_sequence[i - 1]
        pos_prev = pose_prev[:3]
        quat_prev_xyzw = pose_prev[3:]

        r_prev = R.from_quat(quat_prev_xyzw)
        R_prev_T = r_prev.as_matrix().T
        t_prev_inv = -R_prev_T @ pos_prev

        T_prev_inv = np.eye(4)
        T_prev_inv[:3, :3] = R_prev_T
        T_prev_inv[:3, 3] = t_prev_inv

        pose_curr = pose_sequence[i]
        pos_curr = pose_curr[:3]
        quat_curr_xyzw = pose_curr[3:]

        T_curr = np.eye(4)
        T_curr[:3, :3] = R.from_quat(quat_curr_xyzw).as_matrix()
        T_curr[:3, 3] = pos_curr

        delta_T_i = T_prev_inv @ T_curr

        rotation_6d = delta_T_i[:2, :3].flatten()
        translation_3d = delta_T_i[:3, 3]
        compressed_T = np.concatenate([rotation_6d, translation_3d])

        delta_T_list.append(compressed_T)

    return np.array(delta_T_list)


def dual_arm_poses_to_relative(dual_pose_sequence):
    num_poses = dual_pose_sequence.shape[0]

    pose_seq_arm1 = dual_pose_sequence[:, 0:7]
    gripper_seq_arm1 = dual_pose_sequence[:, 7]

    pose_seq_arm2 = dual_pose_sequence[:, 8:15]
    gripper_seq_arm2 = dual_pose_sequence[:, 15]

    # 分别计算两个臂
    delta_T_seq_arm1 = poses_to_relative_matrices(pose_seq_arm1)
    delta_T_seq_arm2 = poses_to_relative_matrices(pose_seq_arm2)
    # delta_T_seq_arm1 = poses_to_relative_matrices_adjacent(pose_seq_arm1) # adjacent frames
    # delta_T_seq_arm2 = poses_to_relative_matrices_adjacent(pose_seq_arm2)

    delta_T_flat_arm1 = delta_T_seq_arm1.reshape(num_poses, 9)
    delta_T_flat_arm2 = delta_T_seq_arm2.reshape(num_poses, 9)

    gripper_arm1 = gripper_seq_arm1.reshape(-1, 1)
    gripper_arm2 = gripper_seq_arm2.reshape(-1, 1)

    final_output = np.concatenate([
        delta_T_flat_arm1,
        gripper_arm1,
        delta_T_flat_arm2,
        gripper_arm2
    ], axis=1)

    return final_output


def add_rotation_noise(pose_sequence, noise_sigma=0.055, same_noise_for_both_arms=True):
    """
    为一个双臂姿态序列中的旋转部分添加一个统一的、系统性的噪声。

    与之前版本不同，此版本为所有左臂姿态生成一个单一的随机旋转噪声，
    并为所有右臂姿态生成另一个单一的随机旋转噪声。这模拟了系统性误差。

    该函数假定输入数据格式为 (N, 16)，其中 N 是序列长度。
    每一行的16个维度定义如下:
    - cols 0-2:  左臂末端位置 (x, y, z)
    - cols 3-6:  左臂末端姿态四元数 (x, y, z, w)
    - col 7:     左臂夹爪状态
    - cols 8-10: 右臂末端位置 (x, y, z)
    - cols 11-14:右臂末端姿态四元数 (x, y, z, w)
    - col 15:    右臂夹爪状态

    参数:
    pose_sequence (np.ndarray): 形状为 (N, 16) 的双臂姿态序列。
    noise_sigma (float): 旋转向量分量的标准差，用于控制噪声强度。
    same_noise_for_both_arms (bool): 如果为True，则左右臂使用完全相同的噪声。
                                     默认为False，即左右臂的系统噪声是独立的。

    返回:
    np.ndarray: 添加噪声后的新姿态序列，形状仍为 (N, 16)。
    """
    # 0. 输入验证
    if not isinstance(pose_sequence, np.ndarray) or pose_sequence.ndim != 2 or pose_sequence.shape[1] != 16:
        raise ValueError("输入必须是形状为 (N, 16) 的 NumPy 数组。")

    # 1. 创建一个副本
    noisy_sequence = pose_sequence.copy()

    # 2. 提取左右臂的原始四元数 (批处理)
    q_left_orig_batch = pose_sequence[:, 3:7]
    q_right_orig_batch = pose_sequence[:, 11:15]

    # 3. 【核心改动】为整个序列生成一个单一的随机旋转向量
    #    不再是 (N, 3)，而是 (3,)
    noise_vec_left = np.random.normal(0, noise_sigma, 3)
    if same_noise_for_both_arms:
        noise_vec_right = noise_vec_left
    else:
        noise_vec_right = np.random.normal(0, noise_sigma, 3)

    # 4. 将向量和四元数转换为 Rotation 对象
    #    q_noise_* 是单个旋转, q_orig_* 是包含N个旋转的批次
    q_noise_left = R.from_rotvec(noise_vec_left)
    q_noise_right = R.from_rotvec(noise_vec_right)

    q_orig_left = R.from_quat(q_left_orig_batch)
    q_orig_right = R.from_quat(q_right_orig_batch)

    # 5. 【核心改动】复合旋转
    #    Scipy的广播机制会自动将单个的 q_noise_* 应用到批次 q_orig_* 中的每一个元素
    q_new_left = q_noise_left * q_orig_left
    q_new_right = q_noise_right * q_orig_right

    # 6. 将结果转换回 (N, 4) 的四元数数组
    q_new_left_xyzw = q_new_left.as_quat()
    q_new_right_xyzw = q_new_right.as_quat()

    # 7. 将加噪后的四元数放回副本数组
    noisy_sequence[:, 3:7] = q_new_left_xyzw
    noisy_sequence[:, 11:15] = q_new_right_xyzw

    return noisy_sequence


if __name__ == "__main__":
    pos0 = np.array([2.0, 3.0, 1.5])
    r0 = R.from_euler('z', 45, degrees=True)

    pos1 = np.array([3, 4, 2])
    r1 = R.from_euler('x', 50, degrees=True)

    # 分别计算
    R0_T = r0.as_matrix().T
    t0_inv = -R0_T @ pos0

    T0_inv = np.eye(4)
    T0_inv[:3, :3] = R0_T
    T0_inv[:3, 3] = t0_inv

    # 矩阵求逆
    T0 = np.eye(4)
    T0[:3, :3] = r0.as_matrix()
    T0[:3, 3] = pos0
    T0_inv_direct = np.linalg.inv(T0)

    # print("T0_inv", T0_inv)
    # print("T0_inv_direct: ", T0_inv_direct)

    # 第0帧相对于第i帧的旋转
    R_0 = r0.as_matrix()
    R_1 = r1.as_matrix()
    R_rel = R_0.T @ R_1
    print("R_rel[:2, :3]: ", R_rel[:2, :3])

    # 第i帧相对于第0帧的旋转
    T1 = np.eye(4)
    T1[:3, :3] = r1.as_matrix()
    T1[:3, 3] = pos1
    delta_T_i = T0_inv @ T1
    print("delta_T_i[:2, :3]: ", delta_T_i[:2, :3])
