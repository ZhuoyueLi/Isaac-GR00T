import os
import argparse

def calculate_success_rates(root_path):
    task_success_counts = {}
    task_failure_counts = {}

    for task in os.listdir(root_path):
        task_path = os.path.join(root_path, task)
        if not os.path.isdir(task_path) or task.startswith("."):
            continue  # Skip non-directory files and hidden folders
        
        success_count = 0
        failure_count = 0

        for global_task in os.listdir(task_path):
            global_task_path = os.path.join(task_path, global_task)
            if not os.path.isdir(global_task_path):
                continue

            for subsequence in os.listdir(global_task_path):
                subsequence_path = os.path.join(global_task_path, subsequence)
                if not os.path.isdir(subsequence_path):
                    continue

                if "success" in os.listdir(subsequence_path):
                    success_count += 1
                elif "failure" in os.listdir(subsequence_path):
                    failure_count += 1
        
        task_success_counts[task] = success_count
        task_failure_counts[task] = failure_count
    
    # Compute success rates
    success_rates = {}
    for task in task_success_counts:
        total_attempts = task_success_counts[task] + task_failure_counts[task]
        if total_attempts > 0:
            success_rates[task] = task_success_counts[task] / total_attempts
        else:
            success_rates[task] = None  # No data available for this task
    
    # Print results
    print("Task Success Rates:")
    for task, rate in success_rates.items():
        rate_str = f"{rate:.2%}" if rate is not None else "No data"
        print(f"{task}: {rate_str} ({task_success_counts[task]} success, {task_failure_counts[task]} failure)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate success rates per task.")
    parser.add_argument("path", type=str, help="Path to the root directory containing tasks.")
    args = parser.parse_args()
    
    calculate_success_rates(args.path)