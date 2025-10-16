# Testing Directory

## 📂 Contents

This directory contains test scripts and utilities for GENESIS.

### 🧪 Test Scripts

- **`test_config.py`** - Configuration system tests
  - Tests path resolution (absolute, relative, home directory)
  - Tests YAML config loading
  - Validates config parsing and type conversion
  
  **Usage:**
  ```bash
  cd /home/work/GENESIS/GENESIS-pmj0324/GENESIS
  python3 testing/test_config.py
  ```

### 📝 Test Categories

- **Unit Tests** - Individual component testing
- **Integration Tests** - End-to-end workflow testing
- **Config Tests** - Configuration system validation

### ✅ Running Tests

```bash
# Run config tests
python3 testing/test_config.py

# Add more test scripts here as needed
```

### 🎯 Adding New Tests

1. Create test script in this directory
2. Follow naming convention: `test_*.py`
3. Document in this README
4. Ensure tests are self-contained and can run independently


