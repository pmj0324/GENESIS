# Config System Changelog

## 🎉 Version 2.0 - YAML-Relative Path Support (2025-01-16)

### ✨ New Features

1. **YAML-Relative Path Resolution**
   - Paths in YAML files are now resolved relative to the YAML file location
   - Supports absolute paths, home directory (`~`), and relative paths
   - Automatically resolves paths during config loading

2. **Simplified Path Handling**
   - Removed Git repository auto-detection
   - No more environment variable dependencies (`${GENESIS_ROOT}`)
   - Cleaner, more predictable path resolution

### 📝 Code Improvements

1. **Reduced Complexity**
   - Removed 40+ lines of Git repository detection code
   - Simplified `__post_init__` methods across all config classes
   - Added helper functions: `resolve_path()`, `convert_to_type()`, `convert_to_tuple()`

2. **Better Type Conversion**
   - Unified type conversion logic
   - Cleaner boolean handling
   - More robust string/null handling

3. **Enhanced Logging**
   - Config loading now prints resolved paths
   - Easy to verify path resolution
   - Better debugging experience

### 📊 Metrics

- **Code Reduction**: 559 → 553 lines (-6 lines, but much cleaner)
- **Complexity**: Significantly reduced
- **Maintainability**: Greatly improved
- **Test Coverage**: Added comprehensive tests

### 🔄 Migration Guide

#### Before (v1.0)
```yaml
# Old way: Environment variables
data:
  h5_path: "${GENESIS_ROOT}GENESIS-data/train.h5"
```

#### After (v2.0)
```yaml
# New way: YAML-relative paths
data:
  h5_path: "../GENESIS-data/train.h5"
```

**Migration Steps:**
1. Open your YAML config files
2. Replace `${GENESIS_ROOT}...` with relative paths
3. Test with `python3 test_config.py`

### 📖 Examples

#### Example 1: Basic Relative Path
```yaml
# configs/default.yaml
data:
  h5_path: "../GENESIS-data/train.h5"  # One level up from configs/
```

#### Example 2: Nested Config
```yaml
# configs/experiments/my_exp.yaml
data:
  h5_path: "../../GENESIS-data/train.h5"  # Two levels up
```

#### Example 3: Absolute Path (still supported)
```yaml
# configs/production.yaml
data:
  h5_path: "/mnt/data/icecube/train.h5"  # Absolute path
```

### 🧪 Testing

Added comprehensive test suite:
```bash
cd /home/work/GENESIS/GENESIS-pmj0324/GENESIS
python3 test_config.py
```

**Test Coverage:**
- ✅ Absolute path resolution
- ✅ Home directory expansion
- ✅ YAML-relative path resolution
- ✅ Config loading from YAML
- ✅ Multiple path scenarios

### 🐛 Bug Fixes

1. **Fixed**: Git not found error on systems without Git
2. **Fixed**: Inconsistent path resolution with environment variables
3. **Fixed**: Complex type conversion causing YAML parsing issues

### ⚠️ Breaking Changes

1. **Environment Variables No Longer Supported**
   - `${GENESIS_ROOT}` no longer works
   - Use YAML-relative paths instead
   - Migration is straightforward (see above)

2. **Git Repository Detection Removed**
   - No automatic Git repository detection
   - Paths are always relative to YAML file
   - More predictable behavior

### 📚 New Documentation

- [CONFIG_PATH_RESOLUTION.md](docs/guides/CONFIG_PATH_RESOLUTION.md) - Complete guide
- `test_config.py` - Test suite and examples

---

## 📦 Version 1.0 - Initial Release

### Features

1. **Dataclass-based Configuration**
   - ModelConfig, DiffusionConfig, DataConfig, TrainingConfig
   - Type-safe configuration management

2. **YAML Configuration Files**
   - Human-readable configuration
   - Easy experimentation

3. **Environment Variable Support**
   - `${GENESIS_ROOT}` for project root
   - Git repository auto-detection

4. **Type Conversion**
   - Automatic type conversion from YAML
   - Handles strings, bools, floats, ints

### Known Issues

1. Complex Git repository detection (now removed)
2. Environment variable dependency (now removed)
3. Long `__post_init__` methods (now improved)

---

## 🔮 Future Plans

### Version 2.1 (Planned)
- [ ] Add validation for config values
- [ ] Better error messages for invalid paths
- [ ] Config schema validation
- [ ] Auto-completion support for IDEs

### Version 3.0 (Future)
- [ ] Hierarchical config inheritance
- [ ] Config templates
- [ ] Runtime config updates
- [ ] Web-based config editor

---

## 🙏 Acknowledgments

Thanks to all contributors and users for feedback that led to these improvements!

---

## 📞 Support

- **Issues**: Open a GitHub issue
- **Questions**: See [docs/README.md](docs/README.md)
- **Examples**: Check `test_config.py`

