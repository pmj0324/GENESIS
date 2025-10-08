#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_pmt_dit.py

- PMTSignalsH5 데이터셋 로딩
- PMTDit + GaussianDiffusion 학습
- NaN/Inf 디버깅 도구 포함:
  * 입력(raw), q_sample(x_t), 모델 출력(eps_hat), loss, grad 순서로 점검
  * step 번호와 에폭 진행 퍼센트 출력
"""

import os
import torch
import torch.nn as nn

# === 프로젝트 모듈 import ===
from dataloader.pmt_dataloader import make_dataloader
from models.pmt_dit import PMTDit, GaussianDiffusion, DiffusionConfig


# =========================
# 디버그 유틸
# =========================
def tensor_stats(name: str, x: torch.Tensor, max_print: int = 0):
    """텐서 기본 통계 + NaN/Inf 개수 출력"""
    n_tot = x.numel()
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    finite = torch.isfinite(x)
    msg = f"[{name}] shape={tuple(x.shape)} | nan={n_nan} inf={n_inf} / {n_tot}"
    if finite.any():
        xf = x[finite]
        msg += f" | min={xf.min().item():.4g} max={xf.max().item():.4g} mean={xf.mean().item():.4g} std={xf.std().item():.4g}"
    print(msg)
    if max_print > 0:
        with torch.no_grad():
            flat = x.reshape(-1)
            print(f"  sample values: {flat[:max_print].tolist()}")

def assert_finite(name: str, x: torch.Tensor):
    """비유한 값이 있으면 통계 찍고 바로 에러"""
    if not torch.isfinite(x).all():
        tensor_stats(name, x)
        raise RuntimeError(f"Non-finite values detected in {name}")

def register_nan_hooks(model: nn.Module):
    """주요 모듈의 입/출력에 NaN이 생기면 바로 중단"""
    def _hook(mod, inp, out):
        def _chk(t, tag):
            if isinstance(t, torch.Tensor) and not torch.isfinite(t).all():
                print(f"[NaNHook] {mod.__class__.__name__} {tag} non-finite")
                tensor_stats(f"{mod.__class__.__name__}:{tag}", t)
                raise RuntimeError(f"Non-finite in {mod.__class__.__name__} {tag}")
        if isinstance(inp, tuple):
            for t in inp: _chk(t, "input")
        else:
            _chk(inp, "input")
        if isinstance(out, tuple):
            for t in out: _chk(t, "output")
        else:
            _chk(out, "output")

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.MultiheadAttention)):
            m.register_forward_hook(_hook)


def main():
    # ------------------------
    # 기본 설정
    # ------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h5_path = "/home/work/GENESIS/GENESIS-data/22644_0921.h5"
    batch_size = 8
    num_epochs = 5
    lr = 2e-4

    # ------------------------
    # DataLoader
    # ------------------------
    loader = make_dataloader(
        h5_path=h5_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        replace_time_inf_with=0.0,   # time의 inf를 0.0으로 치환 (원인 파악 후 조정 가능)
        channel_first=True,
    )

    # ------------------------
    # Model + Diffusion 래퍼
    # ------------------------
    L = 5160
    model = PMTDit(
        seq_len=L,
        hidden=512,
        depth=8,
        heads=8,
        dropout=0.1,
        fusion="FiLM",   # or "SUM"
        label_dim=6,
        t_embed_dim=128,
    ).to(device)

    diffusion = GaussianDiffusion(
        model, DiffusionConfig(timesteps=1000, objective="eps")
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # 디버깅 옵션(필요할 때 주석 해제)
    # torch.autograd.set_detect_anomaly(True)
    # register_nan_hooks(model)

    # ------------------------
    # 학습 루프 (NaN 디버깅 포함)
    # ------------------------
    steps_per_epoch = len(loader)
    global_step = 0

    for epoch in range(num_epochs):
        for step_in_epoch, (x_sig, geom, label, idx) in enumerate(loader, start=1):
            # geom shape 맞추기
            if geom.ndim == 2:  # (3,L)
                geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)

            x_sig = x_sig.to(device)    # (B,2,L)
            geom  = geom.to(device)     # (B,3,L)
            label = label.to(device)    # (B,6)

            # 1) 입력 원천 점검
            try:
                assert_finite("x_sig(raw)", x_sig)
                assert_finite("geom(raw)", geom)
                assert_finite("label(raw)", label)
            except RuntimeError:
                print(f"[BAD INPUT] epoch={epoch+1}, step={step_in_epoch}/{steps_per_epoch} idx={idx.tolist()}")
                raise

            # 2) q_sample / 모델 출력 점검 (forward만)
            with torch.no_grad():
                B = x_sig.size(0)
                t = torch.randint(0, diffusion.cfg.timesteps, (B,), device=device, dtype=torch.long)
                x_sig_t = diffusion.q_sample(x_sig, t)
                assert_finite("x_sig_t", x_sig_t)

                eps_hat = model(x_sig_t, geom, t, label)
                assert_finite("eps_hat(fwd-only)", eps_hat)

            # 3) 실제 loss 계산/역전파 (여기서 NaN이면 원인 더 출력)
            loss = diffusion.loss(x_sig, geom, label)
            if not torch.isfinite(loss):
                print("\n[WARN] Non-finite loss detected!")
                print(f"  epoch={epoch+1}, step={step_in_epoch}/{steps_per_epoch}, idx={idx.tolist()}")
                tensor_stats("loss", loss)
                tensor_stats("x_sig(raw)", x_sig)
                tensor_stats("geom(raw)", geom)
                tensor_stats("label(raw)", label)
                # 필요시 문제 배치 저장
                # torch.save({"x_sig": x_sig.cpu(), "geom": geom.cpu(), "label": label.cpu()}, "bad_batch.pt")
                raise RuntimeError("Non-finite loss")

            optimizer.zero_grad()
            loss.backward()

            # 4) 그래디언트 점검
            bad_grad = None
            for n, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    bad_grad = n
                    tensor_stats(f"grad:{n}", p.grad)
                    break
            if bad_grad is not None:
                print(f"[BAD GRAD] epoch={epoch+1}, step={step_in_epoch}/{steps_per_epoch}, param={bad_grad}")
                raise RuntimeError("Non-finite gradient")

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            global_step += 1
            # 진행률(퍼센트) 계산
            pct = 100.0 * step_in_epoch / steps_per_epoch
            if global_step % 50 == 0:
                print(f"[epoch {epoch+1}/{num_epochs}] "
                      f"step {step_in_epoch}/{steps_per_epoch} ({pct:5.1f}%) "
                      f"| loss = {loss.item():.6f}")

    # ------------------------
    # 샘플링 예시
    # ------------------------
    with torch.no_grad():
        # 첫 배치에서 조건 하나 뽑기
        x_sig, geom, label, idx = next(iter(loader))
        if geom.ndim == 2:
            geom = geom.unsqueeze(0).expand(x_sig.size(0), -1, -1)

        geom  = geom[:2].to(device)   # (B=2,3,L)
        label = label[:2].to(device)  # (B=2,6)

        samples = diffusion.sample(label, geom, shape=(2, 2, L))
        print("샘플 shape:", samples.shape)  # (2,2,5160)


if __name__ == "__main__":
    main()