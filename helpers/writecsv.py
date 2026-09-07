"""Write the 150-column CSV and the training npz."""
import numpy as np
import dm_common as C
from dm_common import qmul, qconj, quat_to_axis_angle, qnorm


def unwrap_pelvis_rot(root_q):
    """Recover the continuous per-axis angle mma.cpp stores at columns 141-143."""
    R = C.qmat(qnorm(root_q))
    # Rz(c)*Ry(b)*Rx(a): standard ZYX extraction
    sy = -R[:, 2, 0]
    b = np.arcsin(np.clip(sy, -1.0, 1.0))
    cb = np.cos(b)
    near = np.abs(cb) < 1e-6
    a = np.where(near, np.arctan2(-R[:, 1, 2], R[:, 1, 1]),
                 np.arctan2(R[:, 2, 1], R[:, 2, 2]))
    c = np.where(near, 0.0, np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    return np.stack([np.unwrap(a), np.unwrap(b), np.unwrap(c)], axis=1)


def write(path, joint_q, root_p, root_q, all_keys=True):
    """Bake to CSV. joint_q (F,13,4), root_p (F,3), root_q (F,4)."""
    F = joint_q.shape[0]
    joint_q, root_q = qnorm(joint_q), qnorm(root_q)
    link_p, _ = C.forward_kinematics(root_q, root_p, joint_q)
    com = C.com_of(link_p)

    # joint angular velocity: mma.cpp finite-differences the AXIS-ANGLE target, not the
    aa = quat_to_axis_angle(joint_q)                       # (F,13,3)
    av = np.zeros_like(aa)
    av[1:] = (aa[1:] - aa[:-1]) * C.FPS

    # root linear + angular velocity, same finite-difference convention
    rlv = np.zeros((F, 3))
    rlv[1:] = (root_p[1:] - root_p[:-1]) * C.FPS
    rav = np.zeros((F, 3))
    dq = qmul(root_q[1:], qconj(root_q[:-1]))
    dq = np.where(dq[:, :1] < 0, -dq, dq)
    rav[1:] = quat_to_axis_angle(dq) * C.FPS

    prot = unwrap_pelvis_rot(root_q)
    flag = np.ones((F, 1)) if all_keys else np.zeros((F, 1))

    out = np.concatenate([
        joint_q.reshape(F, 52), link_p.reshape(F, 42), av.reshape(F, 39),
        com, flag, root_q, prot, rlv, rav,
    ], axis=1)
    assert out.shape[1] == 150, out.shape
    np.savetxt(path, out, delimiter=",", fmt="%.6g")
    return out


def write_npz(path, csv_rows):
    """Same derivation animate.py does, so a clip can go straight to training."""
    F = csv_rows.shape[0]
    d = csv_rows[:, :136]
    np.savez(path,
             joint_q=d[:, 0:52].reshape(F, 13, 4),
             link_p=d[:, 52:94].reshape(F, 14, 3),
             joint_av=d[:, 94:133].reshape(F, 13, 3),
             com=d[:, 133:136].reshape(F, 3),
             root_vel=csv_rows[:, 144:147],
             root_ang_vel=csv_rows[:, 147:150],
             root_q=csv_rows[:, 137:141])
