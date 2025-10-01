# render_utils.py
# Utilities for PyTorch3D rendering and compositing.

import torch
import numpy as np
import cv2
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras,
    RasterizationSettings, MeshRenderer, MeshRasterizer,
    SoftPhongShader, PointLights, TexturesVertex
)
from pytorch3d.io import load_objs_as_meshes

# --------- Cameras ---------
def build_pytorch3d_camera_from_KRT(K, R, T, image_size):
    """
    Create a PyTorch3D PerspectiveCameras from real intrinsics/extrinsics.
    Args:
        K: (3,3)
        R: (3,3) world-to-camera or camera-to-world? -> We want camera-to-world in PyTorch3D
           PyTorch3D expects R and T that transform points from world to camera: X_cam = R @ X_world + T
        T: (3,)
        image_size: (H, W)
    Returns:
        cameras: PerspectiveCameras on cuda/cpu
    """
    H, W = image_size
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # PyTorch3D normalizes principal point by image size
    # and expects focal length in pixels (works with PerspectiveCameras with in_ndc=False).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    R_t = torch.tensor(R, dtype=torch.float32, device=device).unsqueeze(0)
    T_t = torch.tensor(T, dtype=torch.float32, device=device).unsqueeze(0)

    cameras = PerspectiveCameras(
        focal_length = torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
        principal_point = torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
        image_size = torch.tensor([[H, W]], dtype=torch.float32, device=device),
        R = R_t,
        T = T_t,
        in_ndc = False, # Use real pixel intrinsics
        device = device
    )
    return cameras

# --------- Simple Meshes ---------
def make_colored_cube(side=0.05, center=(0,0,0)):
    """
    Create a unit cube mesh centered at 'center' with given side length.
    """
    cx, cy, cz = center
    s = side / 2.0
    # 8 vertices
    verts = torch.tensor([
        [cx - s, cy - s, cz - s],
        [cx + s, cy - s, cz - s],
        [cx + s, cy + s, cz - s],
        [cx - s, cy + s, cz - s],
        [cx - s, cy - s, cz + s],
        [cx + s, cy - s, cz + s],
        [cx + s, cy + s, cz + s],
        [cx - s, cy + s, cz + s],
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0,1,2], [0,2,3],  # back
        [4,5,6], [4,6,7],  # front
        [0,1,5], [0,5,4],  # bottom
        [2,3,7], [2,7,6],  # top
        [1,2,6], [1,6,5],  # right
        [0,3,7], [0,7,4],  # left
    ], dtype=torch.int64)

    # vertex colors
    colors = torch.tensor([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [1,1,0],
        [1,0,1],
        [0,1,1],
        [0.5,0.5,0.5],
        [1,1,1],
    ], dtype=torch.float32)

    textures = TexturesVertex(verts_features=colors.unsqueeze(0))
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh

# --------- Renderer ---------
def build_renderer(cameras, image_size, raster_blur_radius=0.0, faces_per_pixel=1):
    device = cameras.device
    H, W = image_size
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=raster_blur_radius,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=True
    )
    lights = PointLights(device=device, location=[[0.0, 0.0, 0.3]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer

# --------- Compositing ---------
def composite_on_image(render_rgb, real_bgr, alpha=None, keep_background=True):
    """
    Composite the rendered RGB on top of the real image.
    Args:
        render_rgb: (H,W,3) float32 [0,1]
        real_bgr: (H,W,3) uint8 (BGR)
        alpha: optional (H,W) [0,1] alpha map. If None, compute from non-black render
        keep_background: if False, background set to render only region.
    Returns:
        out_bgr: uint8 composite
    """
    H, W = real_bgr.shape[:2]
    render_bgr = (render_rgb[..., ::-1] * 255.0).clip(0,255).astype(np.uint8)

    if alpha is None:
        # Alpha where render is non-black
        alpha = (render_bgr.sum(axis=-1) > 0).astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (5,5), 0)

    alpha3 = np.dstack([alpha, alpha, alpha])
    real = real_bgr.astype(np.float32)
    rend = render_bgr.astype(np.float32)
    comp = alpha3 * rend + (1 - alpha3) * real
    return comp.clip(0,255).astype(np.uint8)
