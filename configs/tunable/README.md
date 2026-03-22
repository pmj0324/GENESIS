# Tunable Model Configs

These examples use the new `preset + explicit override` model parsing path.

Rules:
- Keep `preset` to start from `S/B/L`.
- Add explicit fields in the same block to override preset values.
- Remove `preset` entirely for a fully custom architecture.

Examples in this folder:
- `dit_diffusion_preset_override.yaml`
- `dit_diffusion_fully_custom.yaml`
- `unet_diffusion_preset_override.yaml`
- `unet_diffusion_fully_custom.yaml`
- `swin_diffusion_preset_override.yaml`
- `swin_diffusion_fully_custom.yaml`

Run example:

```bash
cd GENESIS
python train.py --config configs/tunable/dit_diffusion_preset_override.yaml
```
