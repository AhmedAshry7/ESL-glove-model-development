import os
import subprocess
import time
import sys

def run_step(step_num, total_steps, name, script_path, extra_args=None):
    print(f" [{step_num}/{total_steps}] {name}")

    start_time = time.time()
    try:
        python_exe = sys.executable
        cmd = [python_exe, script_path]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n [ERROR] Step failed: {name} (Exit Code {e.returncode})")
        print("Please check the error logs above.")
        exit(1)
    except FileNotFoundError:
        print(f"\n [ERROR] Script not found: {script_path}")
        exit(1)
        
    duration = time.time() - start_time
    print(f"\n {name} completed in {duration:.1f}s")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(base_dir, "tests", "test_windowed_pipeline.py")
    
    
    run_step(1,4, "Converting to CSV", os.path.join(base_dir, "data", "csv_converter.py"))
    run_step(2,4, "Train Windowed randomforest", os.path.join(base_dir, "train_models", "train_windowed_rf.py"))
    run_step(3,4, "running on validation", test_script, extra_args=["val"])
    run_step(4,4, "running on test", test_script, extra_args=["test"])
    

if __name__ == "__main__":
    main()
