import torch.nn as nn

class SimpleCondUNet(nn.Module):
    def __init__(self, cond_dim=3, time_embed_dim=32, base_ch=32):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.inc = nn.Conv2d(2, base_ch, 3, padding=1)
        self.down1 = nn.Conv2d(base_ch, base_ch*2, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(base_ch*2, base_ch*4, 3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(base_ch*4, base_ch*2, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(base_ch*2, base_ch, 4, stride=2, padding=1)
        self.outc = nn.Conv2d(base_ch, 2, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x, t, c):
        # x: (B,2,60,79), t: (B,), c: (B,3)
        B = x.shape[0]
        t_emb = self.time_embed(t.float().unsqueeze(-1)).view(B, -1, 1, 1)
        c_emb = self.cond_embed(c).view(B, -1, 1, 1)
        x = self.relu(self.inc(x))
        x = x + t_emb + c_emb
        d1 = self.relu(self.down1(x))
        d2 = self.relu(self.down2(d1))
        u1 = self.relu(self.up1(d2)) + d1
        u2 = self.relu(self.up2(u1)) + x
        out = self.outc(u2)
        return out  # (B,2,60,79)
