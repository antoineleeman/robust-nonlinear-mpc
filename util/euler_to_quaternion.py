import numpy as np

def euler_to_quaternion(e):
    """
    Convert ZYX Euler angles (roll, pitch, yaw) to quaternion [w, x, y, z]
    """
    roll, pitch, yaw = e

    cr, sr = np.cos(roll/2),  np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2),   np.sin(yaw/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)  # normalize for safety
