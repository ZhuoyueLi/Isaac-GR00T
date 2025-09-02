import os
import numpy as np
import cv2
import subprocess
import argparse
from collections import defaultdict

def get_subtask_order(task_path):
    """Extracts and sorts subtask names based on their order number."""
    subtasks = []
    for subtask in os.listdir(task_path):
        subtask_path = os.path.join(task_path, subtask)
        if os.path.isdir(subtask_path):
            try:
                name, number = subtask.rsplit('_', 1)
                subtasks.append((int(number), subtask))
            except ValueError:
                pass
    return [subtask[1] for subtask in sorted(subtasks)]

def calculate_success_rates(base_path):
    """Calculates success rates and average sequence length for each task."""
    task_stats = {}
    
    for task in os.listdir(base_path):
        if task.startswith('.'):
            continue
        task_path = os.path.join(base_path, task)
        if not os.path.isdir(task_path):
            continue
        
        sequences = os.listdir(task_path)
        success_counts = defaultdict(int)
        total_counts = defaultdict(int)
        sequence_lengths = []
        
        for seq in sequences:
            seq_path = os.path.join(task_path, seq)
            if not os.path.isdir(seq_path):
                continue
            
            subtasks = get_subtask_order(seq_path)
            valid_subtasks = [s for s in subtasks if os.path.exists(os.path.join(seq_path, s, "success"))]
            
            sequence_lengths.append(len(valid_subtasks))
            
            for subtask in subtasks:
                subtask_path = os.path.join(seq_path, subtask)
                
                if os.path.exists(os.path.join(subtask_path, "success")):
                    success_counts[subtask] += 1
                total_counts[subtask] += 1
        
        avg_success_rate = {
            subtask: (success_counts[subtask] / len(sequences)) * 100
            for subtask in total_counts
        }
        avg_sequence_length = np.mean(sequence_lengths) if sequence_lengths else 0
        
        task_stats[task] = {
            "success_rates": avg_success_rate,
            "average_sequence_length": avg_sequence_length
        }
    
    return task_stats

def merge_videos(task_path, task, base_name):
    """Merges videos of individual subtasks in the correct order using ffmpeg."""
    processed_dir = os.path.join("processed_seqs", base_name, task)
    os.makedirs(processed_dir, exist_ok=True)
    
    for seq in os.listdir(task_path):
        seq_path = os.path.join(task_path, seq)
        if not os.path.isdir(seq_path):
            continue
        
        subtasks = get_subtask_order(seq_path)
        temp_list_file = os.path.join(seq_path, "video_list.txt")
        
        video_files = []
        for subtask in subtasks:
            subtask_path = os.path.join(seq_path, subtask)
            primary_video = os.path.abspath(os.path.join(subtask_path, "primary_camera.mp4"))
            
            if os.path.exists(primary_video):
                video_files.append(f"file '{primary_video}'\n")
        
        if video_files:
            with open(temp_list_file, "w") as f:
                f.writelines(video_files)
            
            output_video = os.path.join(processed_dir, f"{task}_{seq}.mp4")
            subprocess.run([
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", temp_list_file,
                "-c", "copy", output_video
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_list_file)

def main(base_path):
    if os.path.isabs(base_path):
        raise ValueError("Absolute paths are not allowed. Please provide a relative path.")
    
    base_name = os.path.basename(os.path.normpath(base_path))
    success_rates = calculate_success_rates(base_path)
    
    for task, stats in success_rates.items():
        print(f"Task: {task}")
        print(f"  Average Sequence Length: {stats['average_sequence_length']:.2f}")
        print("  Success Rates:")
        for subtask, rate in stats["success_rates"].items():
            print(f"    {subtask}: {rate:.2f}%")
    
    print("\nMerging videos...")
    for task in os.listdir(base_path):
        if task.startswith('.'):
            continue
        task_path = os.path.join(base_path, task)
        if os.path.isdir(task_path):
            merge_videos(task_path, task, base_name)
    print("Video merging completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process success rates and merge videos.")
    parser.add_argument("base_directory", type=str, help="Path to the base directory containing task sequences")
    args = parser.parse_args()
    
    main(args.base_directory)
