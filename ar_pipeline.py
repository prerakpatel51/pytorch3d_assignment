# ar_pipeline.py
# A minimal CLI-like runner that mirrors the notebook steps.
# Usage (after installing deps): python ar_pipeline.py --image path/to.jpg --K "fx,fy,cx,cy" --aruco_id 0

import argparse, cv2, numpy as np, torch
from pose_utils import estimate_pose_from_aruco
from render_utils import build_pytorch3d_camera_from_KRT, make_colored_cube, build_renderer, composite_on_image

def parse_K(s):
    fx, fy, cx, cy = [float(x) for x in s.split(",")]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], dtype=np.float32)
    return K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="Path to the real image")
    ap.add_argument("--K", required=True, help="fx,fy,cx,cy")
    ap.add_argument("--aruco_id", type=int, default=0)
    ap.add_argument("--marker_length", type=float, default=0.04, help="meters")
    args = ap.parse_args()

    K = parse_K(args.K)
    img_bgr = cv2.imread(args.image)
    H, W = img_bgr.shape[:2]

    R, T, _ = estimate_pose_from_aruco(img_bgr, K, None, aruco_id=args.aruco_id, marker_length=args.marker_length)

    cameras = build_pytorch3d_camera_from_KRT(K, R, T, (H, W))
    mesh = make_colored_cube(side=0.05, center=(0,0,0))
    renderer = build_renderer(cameras, (H, W))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = mesh.to(device)
    image = renderer(meshes_world=mesh, cameras=cameras)
    rgb = image[0, ..., :3].detach().cpu().numpy()

    comp = composite_on_image(rgb, img_bgr)
    cv2.imwrite("ar_composite.png", comp)
    print("Saved: ar_composite.png")

if __name__ == "__main__":
    main()
