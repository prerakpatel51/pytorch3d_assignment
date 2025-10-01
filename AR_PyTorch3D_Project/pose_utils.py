# pose_utils.py
# Utilities for camera calibration and planar pose estimation using OpenCV.

import cv2
import numpy as np

# ------------ Calibration (Checkerboard) ------------
def calibrate_from_checkerboard(image_files, board_size=(8,6), square_size=0.024):
    """
    Calibrate camera intrinsics from checkerboard images.
    Args:
        image_files: list of file paths to checkerboard images
        board_size: inner corners (cols, rows) e.g. (8,6)
        square_size: size of a square in meters
    Returns:
        K (3x3), dist_coeffs (k1,k2,p1,p2,k3...), rms, img_size
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_size[1]*board_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []
    imgpoints = []
    img_size = None

    for f in image_files:
        img = cv2.imread(f)
        if img is None:
            print(f"[WARN] Could not read {f}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_size = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
        else:
            print(f"[INFO] Checkerboard not found in {f}")

    if len(objpoints) < 5:
        raise RuntimeError("Need at least 5 valid checkerboard detections for stable calibration.")

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None
    )
    return K, dist, ret, img_size


# ------------ Pose from ArUco ------------
def estimate_pose_from_aruco(image_bgr, K, dist_coeffs, aruco_id=0, marker_length=0.04, dict_name=cv2.aruco.DICT_4X4_50):
    """
    Estimate camera pose wrt an ArUco marker.
    Args:
        image_bgr: input BGR image
        K: camera matrix
        dist_coeffs: distortion coeffs (can be None for zero)
        aruco_id: target marker id
        marker_length: marker side length in meters
        dict_name: cv2.aruco dictionary constant
    Returns:
        R (3x3), T (3,), corners (4x2) in image
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_name)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        raise RuntimeError("No ArUco markers detected.")
    # Find the target ID
    idx = np.where(ids.flatten() == aruco_id)[0]
    if len(idx) == 0:
        raise RuntimeError(f"ArUco id {aruco_id} not found. Detected ids: {ids.flatten()}")
    idx = idx[0]

    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners[idx], marker_length, K, dist_coeffs if dist_coeffs is not None else np.zeros((5,1))
    )
    rvec = rvec.reshape(-1)
    tvec = tvec.reshape(-1)

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, corners[idx].reshape(-1,2)


# ------------ Pose from Planar Homography ------------
def pose_from_planar_homography(H, K, normalize=True):
    """
    Recover [R|t] from homography H (world plane Z=0) and intrinsics K.
    Returns:
        R (3x3), t (3,), normal sign corrected.
    """
    K_inv = np.linalg.inv(K)
    h1 = H[:,0]; h2 = H[:,1]; h3 = H[:,2]
    lam = 1.0 / np.linalg.norm(K_inv @ h1)
    r1 = lam * (K_inv @ h1)
    r2 = lam * (K_inv @ h2)
    r3 = np.cross(r1, r2)
    t  = lam * (K_inv @ h3)
    R  = np.column_stack((r1, r2, r3))
    # Orthonormalize R via SVD
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if normalize and R[2,2] < 0:
        R = -R
        t = -t
    return R, t


def homography_from_points(pts_world, pts_img, K=None):
    """
    Compute homography H such that x_img ~ H x_world (homog).
    If K is provided and points are normalized, this is DLT in pixel coords.
    pts_world: Nx2
    pts_img: Nx2
    """
    assert pts_world.shape[0] >= 4 and pts_img.shape[0] >= 4
    N = pts_world.shape[0]
    A = []
    for i in range(N):
        X, Y = pts_world[i]
        x, y = pts_img[i]
        A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
        A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1, :]
    H = h.reshape(3,3)
    return H / H[2,2]
