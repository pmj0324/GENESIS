#!/usr/bin/env python3
"""
Test script for config.py path resolution
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config_from_file, resolve_path


def test_path_resolution():
    """Test path resolution functionality"""
    print("="*70)
    print("Testing Path Resolution")
    print("="*70)
    
    yaml_dir = Path("/home/work/GENESIS/GENESIS-pmj0324/GENESIS/configs")
    
    # Test 1: Absolute path
    abs_path = "/absolute/path/data.h5"
    resolved = resolve_path(abs_path, yaml_dir)
    print(f"\n1. Absolute path:")
    print(f"   Input:    {abs_path}")
    print(f"   Resolved: {resolved}")
    assert resolved == abs_path, "Absolute path should remain unchanged"
    
    # Test 2: Home directory
    home_path = "~/data/train.h5"
    resolved = resolve_path(home_path, yaml_dir)
    print(f"\n2. Home directory path:")
    print(f"   Input:    {home_path}")
    print(f"   Resolved: {resolved}")
    assert resolved.startswith("/home/"), "Home path should be expanded"
    
    # Test 3: Relative path
    rel_path = "../GENESIS-data/22644_0921_time_shift.h5"
    resolved = resolve_path(rel_path, yaml_dir)
    print(f"\n3. Relative path:")
    print(f"   Input:    {rel_path}")
    print(f"   YAML dir: {yaml_dir}")
    print(f"   Resolved: {resolved}")
    expected = str((yaml_dir / rel_path).resolve())
    assert resolved == expected, f"Expected {expected}, got {resolved}"
    
    # Test 4: Relative path without yaml_dir (uses cwd)
    rel_path2 = "data/train.h5"
    resolved = resolve_path(rel_path2)
    print(f"\n4. Relative path (no yaml_dir):")
    print(f"   Input:    {rel_path2}")
    print(f"   Resolved: {resolved}")
    
    print(f"\n{'='*70}")
    print("✅ All path resolution tests passed!")
    print("="*70)


def test_config_loading():
    """Test loading config from YAML"""
    print("\n" + "="*70)
    print("Testing Config Loading")
    print("="*70)
    
    config_path = "configs/default.yaml"
    print(f"\nLoading config: {config_path}")
    print("-"*70)
    
    config = load_config_from_file(config_path)
    
    print(f"\n📊 Loaded Configuration:")
    print(f"   Experiment: {config.experiment_name}")
    print(f"   Data path:  {config.data.h5_path}")
    print(f"   Batch size: {config.data.batch_size}")
    print(f"   Epochs:     {config.training.num_epochs}")
    print(f"   Model:      {config.model.architecture if hasattr(config.model, 'architecture') else 'dit'}")
    print(f"   Hidden:     {config.model.hidden}")
    print(f"   Depth:      {config.model.depth}")
    
    print(f"\n{'='*70}")
    print("✅ Config loading test passed!")
    print("="*70)
    
    return config


def test_relative_paths():
    """Test different path scenarios"""
    print("\n" + "="*70)
    print("Testing Different Path Scenarios")
    print("="*70)
    
    # Create a temporary test YAML
    test_yaml_content = """
experiment_name: "test"
model:
  seq_len: 5160
  hidden: 64
  depth: 2
diffusion:
  timesteps: 100
data:
  h5_path: "../GENESIS-data/test.h5"  # Relative to YAML
  batch_size: 8
training:
  num_epochs: 1
"""
    
    test_yaml_path = Path("test_config_temp.yaml")
    test_yaml_path.write_text(test_yaml_content)
    
    print(f"\n📝 Created temporary YAML: {test_yaml_path.resolve()}")
    print(f"   Content h5_path: ../GENESIS-data/test.h5")
    
    try:
        config = load_config_from_file(str(test_yaml_path))
        print(f"\n✅ Resolved h5_path: {config.data.h5_path}")
        
        # Verify it's an absolute path
        assert Path(config.data.h5_path).is_absolute(), "Should be absolute path"
        print(f"✅ Path is absolute: {Path(config.data.h5_path).is_absolute()}")
        
    finally:
        # Cleanup
        if test_yaml_path.exists():
            test_yaml_path.unlink()
            print(f"\n🗑️  Cleaned up temporary file")
    
    print(f"\n{'='*70}")
    print("✅ Path scenario tests passed!")
    print("="*70)


if __name__ == "__main__":
    try:
        print("\n" + "🧪 "*35)
        print("GENESIS Config System Tests")
        print("🧪 "*35 + "\n")
        
        # Run tests
        test_path_resolution()
        config = test_config_loading()
        test_relative_paths()
        
        print("\n" + "="*70)
        print("🎉 All tests passed successfully!")
        print("="*70)
        print("\n📝 Summary:")
        print("   ✅ Path resolution works correctly")
        print("   ✅ YAML config loading works")
        print("   ✅ Relative paths resolved to YAML directory")
        print("   ✅ Absolute paths preserved")
        print("   ✅ Home directory expansion works")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

