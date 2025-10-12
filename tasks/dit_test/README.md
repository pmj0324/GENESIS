# GENESIS Task: dit_test

**Created**: 2025-10-13 05:08:09  
**Config**: default.yaml

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
cd tasks/dit_test
./run.sh
```

### Direct Python
```bash
cd /Users/pmj0324/Sicence/IceCube/GENESIS/GENESIS
python3 scripts/train.py --config tasks/dit_test/config.yaml
```

### Server (Background)
```bash
cd tasks/dit_test
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
