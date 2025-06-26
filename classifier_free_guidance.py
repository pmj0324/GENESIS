import torch
import numpy as np

@torch.no_grad()
def sample_ddpm_cfg(
    model,
    shape,
    alpha_bar,
    T,
    device,
    cond,              # (1, 3) or (B, 3)
    cfg_scale=0.5,
    n_steps=50,        # 샘플링 반복수(노이즈 초기화 n_steps회)
    uncond=None        # (1, 3) or (B, 3) 무조건성
):
    ab = torch.from_numpy(alpha_bar).float().to(device)
    samples = []
    for i in range(n_steps):
        x = torch.randn(shape, device=device)
        for t in reversed(range(T)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            # Classifier-Free Guidance
            # 1. Uncond pred
            uncond_pred = model(x, t_tensor, uncond)
            # 2. Cond pred
            cond_pred = model(x, t_tensor, cond)
            # 3. CFG
            pred_noise = uncond_pred + cfg_scale * (cond_pred - uncond_pred)
            sqrt_ab = ab[t] ** 0.5
            sqrt_1m_ab = (1 - ab[t]) ** 0.5
            x0_pred = (x - sqrt_1m_ab * pred_noise) / sqrt_ab
            if t > 0:
                noise = torch.randn_like(x)
                beta_t = 1 - ab[t] / ab[t - 1]
                x = sqrt_ab * x0_pred + (1 - sqrt_ab) * noise
            else:
                x = x0_pred
        samples.append(x.detach().cpu())
    return torch.stack(samples)  # (n_steps, B, 2, 5160)
