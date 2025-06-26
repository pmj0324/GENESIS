import numpy as np

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)

def get_alpha_bar(beta):
    alpha = 1 - beta
    alpha_bar = np.cumprod(alpha)
    return alpha, alpha_bar

def q_sample(x0, t, alpha_bar):
    device = x0.device
    ab = torch.from_numpy(alpha_bar).float().to(device)
    sqrt_alpha_bar = ab[t].view(-1, 1, 1, 1).sqrt()
    sqrt_1m_alpha_bar = (1 - ab[t].view(-1, 1, 1, 1)).sqrt()
    noise = torch.randn_like(x0)
    return sqrt_alpha_bar * x0 + sqrt_1m_alpha_bar * noise, noise
