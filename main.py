
# If you're on Google Colab, run this cell.
def main():
    import sys, subprocess, importlib, os, platform

    def pip_install(pkgs):
        for p in pkgs:
            print("Installing:", p)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])

    # Core packages
    pip_install([
        "opencv-contrib-python==4.10.0.84",
        "numpy",
        "matplotlib",
        "imageio"
    ])

    # Torch & PyTorch3D installer (tries to match CUDA/torch).
    # Colab usually has torch preinstalled. We detect and install a matching PyTorch3D wheel.
    import torch
    print("Torch version:", torch.__version__)
    cuda = torch.version.cuda
    print("CUDA version in torch:", cuda)

    def install_pytorch3d():
        # Mapping common CUDA versions to official wheels
        wheels = {
            "12.1": "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/cu121/py3.10_pyt2.3.0/pytorch3d-0.7.7-cp310-cp310-linux_x86_64.whl",
            "11.8": "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/cu118/py3.10_pyt2.1.0/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl",
            "11.7": "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/cu117/py3.10_pyt2.0.1/pytorch3d-0.7.4-cp310-cp310-linux_x86_64.whl"
        }
        url = None
        if cuda is not None:
            # Pick closest
            if cuda.startswith("12.1"):
                url = wheels["12.1"]
            elif cuda.startswith("11.8"):
                url = wheels["11.8"]
            elif cuda.startswith("11.7"):
                url = wheels["11.7"]
        if url is None:
            print("Falling back to source install (this can take longer).")
            pip_install(["'git+https://github.com/facebookresearch/pytorch3d.git@stable'"])
        else:
            print("Installing PyTorch3D wheel:", url)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", url])

    try:
        import pytorch3d
        print("PyTorch3D already available.")
    except Exception as e:
        print("Installing PyTorch3D...")
        install_pytorch3d()

    print("All installs finished.")

    import os, sys, cv2, json, math, numpy as np, torch, imageio, matplotlib.pyplot as plt
    from google.colab import files

    # Download helper modules from this notebook's GitHub or upload manually.
    # For this scaffold, we let users upload the helper files generated.
    print("Please upload pose_utils.py and render_utils.py from the provided ZIP (or your repo).")

    uploaded = files.upload()  # user uploads two .py files here

    for name in uploaded.keys():
        print("Saved", name)
        if name.endswith(".py"):
            pass

    from importlib import reload
    import pose_utils, render_utils
    reload(pose_utils); reload(render_utils)

    print("CUDA available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    do_calibrate = True  # set False if you already know K
    board_size = (8,6)   # inner corners
    square_size = 0.024  # 24 mm squares -> 0.024 m

    K = None
    dist = None
    img_size = None

    if do_calibrate:
        print("Upload checkerboard images...")
        up = files.upload()
        image_files = [k for k in up.keys() if k.lower().endswith((".jpg",".png",".jpeg"))]
        print("Found", len(image_files), "images")
        K, dist, rms, img_size = pose_utils.calibrate_from_checkerboard(image_files, board_size, square_size)
        print("Calibration RMS:", rms)
        print("K =\\n", K)
        print("dist =", dist.ravel())
    else:
        print("Skipping calibration; you will set K manually in next cell.")
        

    # If you skipped calibration, manually define K here:
    use_manual_K = False
    manual_fx, manual_fy, manual_cx, manual_cy = 1200.0, 1200.0, 640.0, 360.0

    if use_manual_K:
        K = np.array([[manual_fx, 0, manual_cx],
                    [0, manual_fy, manual_cy],
                    [0, 0, 1]], dtype=np.float32)
        dist = np.zeros((5,1), dtype=np.float32)
        img_size = (int(manual_cx*2), int(manual_cy*2))
        print("Manual K set.")
        print(K)


    print("Upload your real image (with ArUco or checkerboard visible).")
    up2 = files.upload()
    img_name = next(iter(up2.keys()))
    img_bgr = cv2.imread(img_name)
    H_img, W_img = img_bgr.shape[:2]
    print("Image size:", (H_img, W_img))
    plt.figure(); plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)); plt.title("Input"); plt.axis("off");



    use_aruco = True
    aruco_id = 0       # set to match the marker ID you printed
    marker_length = 0.04  # 4 cm in meters

    R = None; T = None
    if use_aruco:
        R, T, corners = pose_utils.estimate_pose_from_aruco(img_bgr, K, dist, aruco_id=aruco_id, marker_length=marker_length)
        print("R=\\n", R)
        print("T=", T)
        img_axes = img_bgr.copy()
        # Draw axes for visualization
        axis_len = marker_length*0.5
        axis = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
        rvec, _ = cv2.Rodrigues(R)
        imgpts, _ = cv2.projectPoints(axis, rvec, T, K, dist)
        imgpts = imgpts.reshape(-1,2).astype(int)
        img_axes = cv2.line(img_axes, tuple(imgpts[0]), tuple(imgpts[1]), (0,0,255), 2)
        img_axes = cv2.line(img_axes, tuple(imgpts[0]), tuple(imgpts[2]), (0,255,0), 2)
        img_axes = cv2.line(img_axes, tuple(imgpts[0]), tuple(imgpts[3]), (255,0,0), 2)
        plt.figure(); plt.imshow(cv2.cvtColor(img_axes, cv2.COLOR_BGR2RGB)); plt.title("Pose axes on ArUco"); plt.axis("off");
    else:
        # Example: estimate homography from four known world points on the plane and their image points
        # (You can adapt this section to your checkerboard corners.)
        raise NotImplementedError("Set use_aruco=True or implement homography-based pose here.")

    from render_utils import build_pytorch3d_camera_from_KRT
    cameras = build_pytorch3d_camera_from_KRT(K, R, T, (H_img, W_img))
    cameras

    from render_utils import make_colored_cube, build_renderer, composite_on_image

    mesh = make_colored_cube(side=0.06, center=(0,0,0)).to(device)
    renderer = build_renderer(cameras, (H_img, W_img))
    image = renderer(meshes_world=mesh, cameras=cameras)
    rgb = image[0, ..., :3].detach().cpu().numpy()

    plt.figure(figsize=(8,6)); plt.imshow(rgb); plt.title("Rendered RGB"); plt.axis("off");

    comp = composite_on_image(rgb, img_bgr)
    plt.figure(figsize=(8,6)); plt.imshow(cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)); plt.title("AR Composite"); plt.axis("off");

    out_name = "ar_composite.png"
    cv2.imwrite(out_name, comp)
    print("Saved:", out_name)
    files.download(out_name)
