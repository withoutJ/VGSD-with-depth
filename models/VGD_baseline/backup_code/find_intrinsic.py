import pycolmap
import os

video_dir = "D:\\FYP_Momcilo\Datasets\\GSD-Video-Reflection-Prior\\train\\0483-0"

output_path = os.path.join(video_dir, "colmap_output")
database_path = os.path.join(output_path, "database.db")

# Path to the image directory
image_dir = os.path.join(video_dir, "JPEG_Images")

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

# Initialize COLMAP database
database = pycolmap.Database.connect(database_path)
database.create_tables()

# Extract features from images
feature_extractor = pycolmap.ImageFeatureExtractor()
feature_extractor.options.sift_num_threads = -1  # Use all CPU cores
feature_extractor.options.sift_max_num_features = 8192

# Extract features for all images in directory
for image_name in os.listdir(image_dir):
    if image_name.lower().endswith(('.jpg', '.jpeg')):
        image_path = os.path.join(image_dir, image_name)
        feature_extractor.extract(database, image_path)

# Match features between images
matcher = pycolmap.SequentialMatcher()
matcher.options.num_threads = -1  # Use all CPU cores
matcher.match(database)

# Perform camera calibration
reconstructor = pycolmap.IncrementalMapper()
reconstructor.options.min_model_size = 10
reconstructor.options.init_min_num_inliers = 100

# Run reconstruction
reconstruction = reconstructor.begin_reconstruction(database)
if reconstruction is not None:
    print("Camera intrinsics:")
    for camera in reconstruction.cameras.values():
        print(f"Camera model: {camera.model_name}")
        print(f"Parameters: {camera.params}")
        print(f"Width: {camera.width}, Height: {camera.height}")
else:
    print("Reconstruction failed")


