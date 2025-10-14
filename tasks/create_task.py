#!/usr/bin/env python3
"""
Create a new GENESIS experiment task folder.
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime


def create_task(task_name: str, config_template: str = "default", base_dir: Path = None):
    """
    Create a new experiment task folder with configuration.
    
    Args:
        task_name: Name of the task (required)
        config_template: Config template to use (default: "default")
        base_dir: Base directory for tasks (default: current_dir/tasks)
    """
    # Get base directory
    if base_dir is None:
        base_dir = Path(__file__).parent
    
    # Create task directory path
    task_dir = base_dir / task_name
    
    # Check if task already exists
    if task_dir.exists():
        print(f"❌ Error: Task '{task_name}' already exists!")
        print(f"   Path: {task_dir}")
        print(f"\n💡 Choose a different name or delete the existing task:")
        print(f"   rm -rf {task_dir}")
        return False
    
    # Get project root (two levels up from tasks/)
    project_root = base_dir.parent
    configs_dir = project_root / "configs"
    
    # Find config template
    config_file = configs_dir / f"{config_template}.yaml"
    if not config_file.exists():
        print(f"❌ Error: Config template '{config_template}.yaml' not found!")
        print(f"   Looking in: {configs_dir}")
        print(f"\n💡 Available configs:")
        for cfg in configs_dir.glob("*.yaml"):
            print(f"   - {cfg.stem}")
        return False
    
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"📁 Creating GENESIS Task: {task_name}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create directory structure
    print("✓ Creating directories...")
    task_dir.mkdir(parents=True, exist_ok=False)
    (task_dir / "logs").mkdir()
    (task_dir / "outputs").mkdir()
    (task_dir / "outputs" / "checkpoints").mkdir()
    (task_dir / "outputs" / "samples").mkdir()
    (task_dir / "outputs" / "evaluation").mkdir()
    (task_dir / "outputs" / "plots").mkdir()
    
    # Copy config file
    print(f"✓ Copying configuration: {config_template}.yaml")
    shutil.copy(config_file, task_dir / "config.yaml")
    
    # Create run.sh
    print("✓ Creating run.sh...")
    run_sh_content = f"""#!/bin/bash

# GENESIS Training Script
# Task: {task_name}
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Activate micromamba environment
source ~/GENESIS/micromamba_env.sh
micromamba activate genesis

# Suppress ZMQ warnings (optional)
export PYTHONWARNINGS="ignore"

# Save PID for monitoring
echo $$ > .pid

# Run training (data path from config.yaml)
python3 ../../scripts/train.py \\
    --config config.yaml \\
    "$@"

# Clean up PID file on exit
rm -f .pid
"""
    run_sh_path = task_dir / "run.sh"
    run_sh_path.write_text(run_sh_content)
    run_sh_path.chmod(0o755)  # Make executable
    
    # Create start.sh (nohup wrapper)
    print("✓ Creating start.sh...")
    start_sh_content = f"""#!/bin/bash

# GENESIS Training Starter (Background Mode)
# Task: {task_name}
# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

echo "🚀 Starting GENESIS training in background..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if already running
if [ -f .pid ]; then
    PID=$(cat .pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Training is already running!"
        echo "   PID: $PID"
        echo ""
        echo "💡 To monitor:"
        echo "   ./monitor.sh"
        echo ""
        echo "💡 To stop:"
        echo "   ./stop.sh"
        exit 1
    else
        echo "🧹 Cleaning up stale PID file..."
        rm -f .pid
    fi
fi

# Start in background
nohup ./run.sh > nohup.out 2>&1 &
JOB_PID=$!

# Wait a moment to ensure it started
sleep 2

# Check if still running
if ps -p $JOB_PID > /dev/null 2>&1; then
    echo "✅ Training started successfully!"
    echo ""
    echo "📊 Job Information:"
    echo "   PID: $JOB_PID"
    echo "   Task: {task_name}"
    echo "   Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "📝 Log files:"
    echo "   Console output: nohup.out"
    echo "   Training log: logs/*_training.txt"
    echo ""
    echo "💡 Monitoring commands:"
    echo "   ./monitor.sh              # Show status"
    echo "   ./stop.sh                 # Stop training"
    echo "   tail -f nohup.out         # Watch console output"
    echo "   tail -f logs/*_training.txt  # Watch training log"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
else
    echo "❌ Failed to start training!"
    echo ""
    echo "💡 Check nohup.out for errors:"
    echo "   cat nohup.out"
    exit 1
fi
"""
    start_sh_path = task_dir / "start.sh"
    start_sh_path.write_text(start_sh_content)
    start_sh_path.chmod(0o755)
    
    # Create monitor.sh
    print("✓ Creating monitor.sh...")
    monitor_sh_content = f"""#!/bin/bash

# GENESIS Training Monitor
# Task: {task_name}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 GENESIS Training Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if running
if [ -f .pid ]; then
    PID=$(cat .pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Status: RUNNING"
        echo ""
        echo "📊 Process Information:"
        echo "   PID: $PID"
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers | awk '{{printf "   CPU: %s%%\\n   Memory: %s%%\\n   Runtime: %s\\n   Command: %s\\n", $3, $4, $5, substr($0, index($0,$6))}}'
        echo ""
        
        # GPU info
        if command -v nvidia-smi &> /dev/null; then
            echo "🖥️  GPU Usage:"
            nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | while IFS=, read -r idx name util mem_used mem_total; do
                echo "   GPU $idx: $name"
                echo "   Utilization: $util%"
                echo "   Memory: $mem_used MB / $mem_total MB"
            done
            echo ""
        fi
        
        # Log files
        echo "📝 Recent Log Output:"
        if ls logs/*_training.txt 1> /dev/null 2>&1; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -n 20 logs/*_training.txt | grep -E "Epoch|Loss|Val" || tail -n 10 logs/*_training.txt
        else
            echo "   (No training log found yet)"
        fi
        echo ""
        
        echo "💡 Commands:"
        echo "   tail -f nohup.out              # Watch console output"
        echo "   tail -f logs/*_training.txt    # Watch training log"
        echo "   ./stop.sh                      # Stop training"
        echo "   watch -n 5 ./monitor.sh        # Auto-refresh every 5s"
        
    else
        echo "❌ Status: NOT RUNNING (stale PID file)"
        echo "   Stale PID: $PID"
        echo ""
        echo "💡 Clean up and restart:"
        echo "   rm .pid"
        echo "   ./start.sh"
    fi
else
    echo "❌ Status: NOT RUNNING"
    echo ""
    
    # Check if completed
    if [ -f "logs/*_training.txt" ] && grep -q "Training completed" logs/*_training.txt 2>/dev/null; then
        echo "✅ Training appears to be completed"
        echo ""
        echo "📊 Check results:"
        echo "   cat logs/*_training.txt | tail -n 50"
    else
        echo "💡 To start training:"
        echo "   ./start.sh"
    fi
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""
    monitor_sh_path = task_dir / "monitor.sh"
    monitor_sh_path.write_text(monitor_sh_content)
    monitor_sh_path.chmod(0o755)
    
    # Create stop.sh
    print("✓ Creating stop.sh...")
    stop_sh_content = f"""#!/bin/bash

# GENESIS Training Stopper
# Task: {task_name}

echo "🛑 Stopping GENESIS training..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f .pid ]; then
    PID=$(cat .pid)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Found running process: PID $PID"
        echo ""
        
        # Graceful shutdown
        echo "Sending SIGTERM (graceful shutdown)..."
        kill $PID
        
        # Wait for graceful shutdown
        for i in {{1..10}}; do
            sleep 1
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "✅ Process stopped gracefully"
                rm -f .pid
                exit 0
            fi
            echo -n "."
        done
        echo ""
        
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            echo "⚠️  Process still running, forcing shutdown..."
            kill -9 $PID
            sleep 1
            
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "✅ Process forcefully stopped"
                rm -f .pid
            else
                echo "❌ Failed to stop process!"
                echo "   You may need to manually kill PID $PID"
                exit 1
            fi
        fi
    else
        echo "⚠️  Process not running (stale PID file)"
        rm -f .pid
    fi
else
    echo "⚠️  No PID file found - training not running"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
"""
    stop_sh_path = task_dir / "stop.sh"
    stop_sh_path.write_text(stop_sh_content)
    stop_sh_path.chmod(0o755)
    
    # Create README
    print("✓ Creating README.md...")
    readme_content = f"""# GENESIS Task: {task_name}

**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Config**: {config_template}.yaml

## Description

Add your experiment description here.

## Configuration

Edit `config.yaml` to customize:
- Model architecture
- Training parameters
- Data settings
- Diffusion settings

## Running

### 🚀 Quick Start (Foreground)
```bash
cd {task_dir.relative_to(project_root)}
./run.sh
```

### 🖥️ Server Mode (Background with nohup)
```bash
cd {task_dir.relative_to(project_root)}

# Start training in background
./start.sh

# Monitor training status
./monitor.sh

# Watch logs in real-time
tail -f nohup.out              # Console output
tail -f logs/*_training.txt    # Training log

# Stop training
./stop.sh
```

### 📊 Monitoring Commands

**Check Status:**
```bash
./monitor.sh              # Full status with GPU info
watch -n 5 ./monitor.sh   # Auto-refresh every 5 seconds
```

**Watch Logs:**
```bash
tail -f nohup.out              # Console output
tail -f logs/*_training.txt    # Training log (formatted)
```

**Process Information:**
```bash
cat .pid                  # Get PID
ps -p $(cat .pid) -f     # Process details
nvidia-smi               # GPU usage
```

### 🔧 Direct Python (Advanced)
```bash
cd {project_root}
python3 scripts/train.py --config {task_dir.relative_to(project_root)}/config.yaml
```

## Outputs

- `outputs/checkpoints/`: Saved models
- `outputs/samples/`: Generated samples
- `outputs/evaluation/`: Evaluation results
- `logs/`: Training logs

## Notes

Add any experiment notes here.
"""
    (task_dir / "README.md").write_text(readme_content)
    
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✅ Task Created Successfully!")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print(f"📁 Task Directory: {task_dir}")
    print(f"📄 Configuration: {config_template}.yaml")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🚀 Quick Start (Foreground):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   cd {task_dir.relative_to(project_root)}")
    print(f"   ./run.sh")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🖥️  Server Mode (Background with nohup):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   cd {task_dir.relative_to(project_root)}")
    print(f"   ./start.sh           # Start in background")
    print(f"   ./monitor.sh         # Check status")
    print(f"   ./stop.sh            # Stop training")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 Monitoring:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   ./monitor.sh                  # Show status + GPU info")
    print(f"   watch -n 5 ./monitor.sh       # Auto-refresh every 5s")
    print(f"   tail -f nohup.out             # Console output")
    print(f"   tail -f logs/*_training.txt   # Training log")
    print()
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("🔧 Utilities:")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   vim config.yaml               # Edit configuration")
    print(f"   cat .pid                      # Get process ID")
    print(f"   ps -p $(cat .pid) -f          # Process details")
    print(f"   nvidia-smi                    # GPU usage")
    print()
    print("📋 For more details, see README.md in the task directory")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create a new GENESIS experiment task folder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s my_experiment
  %(prog)s test_run --config testing
  %(prog)s large_model --config default

Available config templates:
  - default: Standard configuration
  - testing: Fast testing with 10%% data
  - cosine: Cosine annealing scheduler
  - (see configs/ directory for more)
        """
    )
    
    parser.add_argument(
        "task_name",
        type=str,
        help="Name of the task (required, must be unique)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Config template to use (default: default)"
    )
    
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Base directory for tasks (default: ./tasks)"
    )
    
    args = parser.parse_args()
    
    # Create task
    success = create_task(
        task_name=args.task_name,
        config_template=args.config,
        base_dir=args.base_dir
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()

