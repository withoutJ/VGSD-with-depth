import os
import subprocess
import numpy as np
import pycolmap
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import argparse

def run_feature_extraction(image_dir, database_path):
    """Run COLMAP feature extraction on the images."""
    cmd = [
        "colmap", "feature_extractor",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--ImageReader.single_camera", "1"
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_exhaustive_matcher(database_path):
    """Run COLMAP exhaustive matcher using the built database."""
    cmd = [
        "colmap", "exhaustive_matcher",
        "--database_path", database_path
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_mapper(database_path, image_dir, output_dir):
    """Run COLMAP mapper to obtain a sparse reconstruction."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "colmap", "mapper",
        "--database_path", database_path,
        "--image_path", image_dir,
        "--output_path", output_dir
    ]
    print("Executing:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def compute_intrinsic_matrix(camera):
    """
    For different camera models, compute the 3x3 intrinsic matrix.
    COLMAP supports various camera models. Here we show an example
    for the PINHOLE and SIMPLE_RADIAL models.
    """
    model_name = str(camera.model).split('.')[-1]  # Convert enum to string
    
    if model_name == "PINHOLE":
        # For PINHOLE model, parameters are: [fx, fy, cx, cy]
        fx, fy, cx, cy = camera.params
    elif "SIMPLE" in model_name:
        # For SIMPLE_RADIAL or SIMPLE_PINHOLE model,
        # parameters are: [f, cx, cy] (and maybe one radial parameter)
        f, cx, cy = camera.params[:3]
        fx, fy = f, f
    elif model_name == "OPENCV":
        # For OPENCV model, parameters are: [fx, fy, cx, cy, k1, k2, p1, p2]
        fx, fy, cx, cy = camera.params[:4]
    elif model_name == "RADIAL":
        # For RADIAL model, parameters are: [f, cx, cy, k1, k2]
        f, cx, cy = camera.params[:3]
        fx, fy = f, f
    else:
        raise ValueError(f"Camera model {model_name} not handled.")
    # Construct the intrinsic matrix
    K = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])
    return K


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run COLMAP reconstruction and extract camera intrinsics.')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Directory containing input images')
    parser.add_argument('--database_path', type=str, default="colmap_database.db",
                        help='Path to the COLMAP database file')
    parser.add_argument('--sparse_dir', type=str, default="sparse_reconstruction",
                        help='Output directory for sparse reconstruction')
    args = parser.parse_args()

    # Paths configuration:
    image_dir = args.image_dir
    database_path = args.database_path
    sparse_dir = args.sparse_dir

    # Step 1: Feature extraction.
    print("Starting COLMAP feature extraction...")
    run_feature_extraction(image_dir, database_path)

    # Step 2: Exhaustive matching.
    print("Starting COLMAP exhaustive matcher...")
    run_exhaustive_matcher(database_path)

    # Step 3: Sparse reconstruction.
    print("Starting COLMAP mapping...")
    run_mapper(database_path, image_dir, sparse_dir)

    # COLMAP's mapper will output one or more models.
    # By default, we assume the first model is in "{sparse_dir}/0"
    model_path = os.path.join(sparse_dir, "0")
    if not os.path.exists(model_path):
        print("Error: Model folder not found:", model_path)
        return

    # Step 4: Load the COLMAP model.
    print("Loading COLMAP model...")
    reconstruction = pycolmap.Reconstruction(model_path)
    cameras = reconstruction.cameras
    images = reconstruction.images
    points3D = reconstruction.points3D

    # Step 5: Retrieve and display the camera intrinsic matrix.
    # (Assuming that a single camera setup is used and hence only one camera in the model.)
    if len(cameras) == 0:
        print("No cameras found in the reconstruction.")
        return

    # If more than one camera is present, print each.
    print("\nEstimated Camera Intrinsic Parameters:")
    for cam_id, camera in cameras.items():
        K = compute_intrinsic_matrix(camera)
        print(f"\nCamera ID: {cam_id}")
        print(" Camera Model:", camera.model)
        print(" Image Size: {} x {}".format(camera.width, camera.height))
        print(" Intrinsic Matrix (K):")
        print(K)

if __name__ == '__main__':
    main()