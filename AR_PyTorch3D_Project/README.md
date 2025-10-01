# Assignment 4 — Augmented Reality with PyTorch3D

This repository contains a **fully working Google Colab notebook** and helper utilities to:
1. Estimate camera pose from a planar surface (checkerboard or ArUco).
2. Render a 3D synthetic object using **PyTorch3D**.
3. Composite the render **on top of a real image** using your estimated camera intrinsics/extrinsics.

---

## Quick Start (Colab)

1. Open `ar_pytorch3d_colab.ipynb` in **Google Colab**.
2. Run **Section 1** to install dependencies (OpenCV, PyTorch3D).
3. In **Section 2**, either:
   - **Calibrate** your camera using a checkerboard (recommended), or
   - **Enter known intrinsics** if you already have them.
4. In **Section 3**, **estimate the pose** of a plane (book cover, laptop keyboard, desk) from your input image using:
   - **ArUco markers** (easiest).
   - OR a **checkerboard**.
5. In **Section 4–6**, set up PyTorch3D cameras with your intrinsics and the estimated **R, T**, load or create a 3D object, and **render**.
6. In **Section 7**, composite the render onto your real image and **export** results.

---

## Repository Contents

- `ar_pytorch3d_colab.ipynb` — the end-to-end notebook you submit.
- `pose_utils.py` — camera calibration and plane-pose estimation helpers (ArUco & checkerboard).
- `render_utils.py` — PyTorch3D renderer setup, mesh creation, and compositing functions.
- `ar_pipeline.py` — a simple Python script that mirrors the notebook steps (CLI-style).
- `assets/` — place any reference images here (e.g., checkerboard PDF, ArUco dictionary preview).

---

## Tips for Best Results

- Use a **large, flat** planar surface with **good lighting**.
- Print an **ArUco marker** or a **checkerboard** (8x6, square size known).
- Keep your camera steady; avoid motion blur.
- If the object looks misaligned:
  - Re-check intrinsics (fx, fy, cx, cy).
  - Confirm the **square size (meters)** for checkerboard/ArUco.
  - Verify that the camera coordinate system matches PyTorch3D's convention in the notebook.

---

## Grading Rubric Mapping

- **Camera Pose Estimation (20 pts)**: Sections 2–3 produce correct `K`, `R`, `T` and show diagnostics.
- **Rendering Setup (25 pts)**: Sections 4–5 set a correct PyTorch3D camera with real intrinsics and image alignment.
- **Synthetic Object Integration (25 pts)**: Sections 5–6 align mesh to the real plane; object scale + pose look natural.
- **Results & Visualization (20 pts)**: Section 7 saves multiple overlays and discusses limitations/improvements.
- **Code Quality & Notebook (10 pts)**: The notebook runs end-to-end in Colab and is well documented.

---

## License

Educational use for coursework. Modify freely and cite sources if you add external assets.
