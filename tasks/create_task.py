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

cd {project_root}

python3 scripts/train.py \\
    --config {task_dir.relative_to(project_root)}/config.yaml \\
    "$@"
"""
    run_sh_path = task_dir / "run.sh"
    run_sh_path.write_text(run_sh_content)
    run_sh_path.chmod(0o755)  # Make executable
    
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

### Using run.sh (Recommended)
```bash
cd {task_dir.relative_to(project_root)}
./run.sh
```

### Direct Python
```bash
cd {project_root}
python3 scripts/train.py --config {task_dir.relative_to(project_root)}/config.yaml
```

### Server (Background)
```bash
cd {task_dir.relative_to(project_root)}
nohup ./run.sh > run.out 2>&1 &

# Check logs
tail -f logs/train_*.log
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
    print("🚀 To start training:")
    print(f"   cd {task_dir.relative_to(project_root)}")
    print(f"   ./run.sh")
    print()
    print("📝 Edit config:")
    print(f"   vim {task_dir}/config.yaml")
    print()
    print("📋 Server (background):")
    print(f"   cd {task_dir.relative_to(project_root)}")
    print(f"   nohup ./run.sh > run.out 2>&1 &")
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

