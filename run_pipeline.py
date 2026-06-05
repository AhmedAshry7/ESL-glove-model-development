import os
import subprocess
import time
import sys

# ── ANSI Color Codes ──
BOLD    = "\033[1m"
GREEN   = "\033[92m"
RED     = "\033[91m"
CYAN    = "\033[96m"
YELLOW  = "\033[93m"
DIM     = "\033[2m"
RESET   = "\033[0m"

def print_header(step_num, total_steps, title):
    print(f"\n{CYAN}{'═'*70}")
    print(f" [{step_num}/{total_steps}] 🚀 {title}")
    print(f"{'═'*70}{RESET}\n")

def run_step(step_num, total_steps, name, script_path, extra_args=None):
    print_header(step_num, total_steps, name)
    
    start_time = time.time()
    try:
        python_exe = sys.executable
        cmd = [python_exe, script_path]
        if extra_args:
            cmd.extend(extra_args)
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n{RED}❌ [ERROR] Step failed: {name} (Exit Code {e.returncode}){RESET}")
        print("Please check the error logs above.")
        exit(1)
    except FileNotFoundError:
        print(f"\n{RED}❌ [ERROR] Script not found: {script_path}{RESET}")
        exit(1)
        
    duration = time.time() - start_time
    print(f"\n{GREEN}✅ {name} completed in {duration:.1f}s{RESET}")

def main():
    print(f"\n{BOLD}{CYAN}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       ESL GLOVE — WINDOWED-RF TRAINING PIPELINE v5              ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{RESET}")
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    test_script = os.path.join(base_dir, "tests", "test_windowed_pipeline.py")
    
    total_steps = 4
    total_start = time.time()
    
    # Step 1: Data Aggregation
    run_step(1, total_steps, "Data Aggregation & Stratification",
             os.path.join(base_dir, "data", "csv_converter.py"))
    
    # Step 2: Train Windowed RF Classifier
    run_step(2, total_steps, "Train Windowed RF Classifier (Feature Pipeline + RF)",
             os.path.join(base_dir, "train_models", "train_windowed_rf.py"))
    
    # Step 3: Validation Set Evaluation
    run_step(3, total_steps, "Validation Set Evaluation",
             test_script, extra_args=["val"])
    
    # Step 4: Test Set Evaluation (Final, Unseen)
    run_step(4, total_steps, "Test Set Evaluation (Final, Unseen)",
             test_script, extra_args=["test"])
    
    total_duration = time.time() - total_start
    
    print(f"\n{BOLD}{GREEN}")
    print("★" * 70)
    print(f"  🎉 FULL PIPELINE COMPLETED IN {total_duration:.1f}s")
    print("★" * 70)
    print(f"{RESET}")
    print(f"  {DIM}Model saved to:  ./models/windowed_rf.joblib{RESET}")
    print(f"  {DIM}To re-evaluate:  python tests/test_windowed_pipeline.py [val|test]{RESET}")
    print(f"  {DIM}Inference test:  python tests/inference_test_windowed.py <json_name>{RESET}")
    print()

if __name__ == "__main__":
    main()
