import os
import cv2
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Extract key frames and copy success/failure files.")
parser.add_argument("source_dir", type=str, help="Path to the source directory")
args = parser.parse_args()

source_dir = args.source_dir

dest_dir = f"processed_{os.path.basename(source_dir)}"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

def extract_frames(video_path, save_path_prefix):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_positions = [0, total_frames - 1, int(0.25 * total_frames), int(0.5 * total_frames), int(0.75 * total_frames)]
    
    for idx, pos in enumerate(frame_positions):
        if pos >= total_frames:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if ret:
            frame_name = ["first", "last", "25pct", "50pct", "75pct"][idx]
            cv2.imwrite(f"{save_path_prefix}_{frame_name}.jpg", frame)
    
    cap.release()

# Walk through the directory structure
for root, dirs, files in os.walk(source_dir):
    if any(fname in ["failure", "success"] for fname in files):
        # Compute relative path
        rel_path = os.path.relpath(root, source_dir)
        new_dir = os.path.join(dest_dir, rel_path)
        os.makedirs(new_dir, exist_ok=True)
        
        # Copy success/failure files
        for fname in ["failure", "success"]:
            file_path = os.path.join(root, fname)
            if os.path.exists(file_path):
                shutil.copy(file_path, os.path.join(new_dir, fname))
        
        # Process video frames
        for cam in ["primary_camera.mp4", "secondary_camera.mp4"]:
            video_path = os.path.join(root, cam)
            if os.path.exists(video_path):
                save_prefix = os.path.join(new_dir, cam.replace(".mp4", ""))
                extract_frames(video_path, save_prefix)

print("Processing complete. Extracted success/failure files and keyframes.")
