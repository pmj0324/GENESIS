"""
GENESIS - Trainer

AdamW + Cosine / Plateau schedule + warmup + early stopping.
Diffusion (.loss) 과 Flow Matching (.loss) 모두 지원.
"""

import math
import random
import sys
import time
from copy import deepcopy
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR, CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Callable, Optional, Union

# loss_fn(model, maps [B,3,256,256], params [B,6]) → scalar tensor
LossFn = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


class _C2OTPrefetcher:
    """
    C²OT 배치를 백그라운드 CUDA 스트림으로 미리 GPU에 올려놓는 prefetcher.
    GPU가 현재 배치를 계산하는 동안 다음 배치를 CPU→GPU 전송.
    """
    def __init__(self, pairs: list, batch_size: int, device: torch.device):
        self.pairs      = pairs
        self.batch_size = batch_size
        self.device     = device
        self.stream     = torch.cuda.Stream(device=device)
        self.n_steps    = len(pairs) // batch_size
        self._next      = None
        self._idx       = 0

    def _load(self, idx: int):
        batch   = self.pairs[idx * self.batch_size : (idx + 1) * self.batch_size]
        x_data  = torch.stack([p["x_data"]  for p in batch])
        x_noise = torch.stack([p["x_noise"] for p in batch])
        theta   = torch.stack([p["theta"]   for p in batch])
        with torch.cuda.stream(self.stream):
            x_data  = x_data.to(self.device, non_blocking=True,
                                 memory_format=torch.channels_last).float()
            x_noise = x_noise.to(self.device, non_blocking=True,
                                  memory_format=torch.channels_last).float()
            theta   = theta.to(self.device, non_blocking=True).float()
        return x_data, x_noise, theta

    def __iter__(self):
        self._idx  = 0
        self._next = self._load(0)          # 첫 배치 prefetch
        return self

    def __next__(self):
        if self._idx >= self.n_steps:
            raise StopIteration
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        x_data, x_noise, theta = self._next
        self._idx += 1
        if self._idx < self.n_steps:        # 다음 배치 미리 올리기
            self._next = self._load(self._idx)
        return x_data, x_noise, theta

    def __len__(self):
        return self.n_steps


class Trainer:
    """
    범용 Trainer. Diffusion / Flow Matching 공용.

    사용 예:
        flow = OTFlowMatching()
        trainer = Trainer(model, flow.loss, schedule="cosine", ckpt_dir="ckpts/")
        trainer.fit(train_loader, val_loader)
    """

    def __init__(
        self,
        model:        nn.Module,
        loss_fn:      LossFn,
        # Optimizer
        lr:           float  = 1e-4,
        weight_decay: float  = 1e-2,
        betas:        tuple  = (0.9, 0.999),
        optimizer:    str    = "adamw",   # "adamw" | "adam" | "sgd"
        momentum:     float  = 0.9,       # SGD 전용
        nesterov:     bool   = True,      # SGD 전용
        # LR Schedule
        schedule:     str    = "cosine",   # "cosine" | "cosine_warmup" | "cosine_restarts" | "plateau"
        warmup_epochs: int   = 5,
        warmup_scale:  float = 1.0,        # warmup 분모 스케일: LR = base_lr * (epoch+1) / (warmup_epochs * warmup_scale)
        # Cosine / cosine_warmup
        T_max:        int    = 200,
        eta_min:      float  = 1e-6,
        # Plateau
        plateau_patience: int   = 5,
        plateau_factor:   float = 0.5,
        # cosine_restarts 전용
        T_0:          int    = 50,
        T_mult:       int    = 2,
        # Training
        max_epochs:   int    = 200,
        grad_clip:    float  = 1.0,
        early_stop_patience: int = 20,
        # Device & Checkpoint
        device:       Union[str, torch.device] = "cuda",
        ckpt_dir:     Optional[Path] = None,
        ckpt_name:    str = "best.pt",
        # Epoch callback (e.g. EpochVisualizer)
        epoch_callback: Optional[Callable] = None,
        # Data info (배너 표시용)
        data_fraction: float = 1.0,
        # C²OT (Condition-Aware OT) — None이면 기존 학습 그대로
        c2ot_sampler = None,   # Optional[C2OTPairSampler]
        c2ot_loss_fn = None,   # Optional[Callable]  — paired_loss(model, x_data, x_noise, cond)
        # Gradient Accumulation
        grad_accum_steps: int = 1,   # 유효 배치 = batch_size × grad_accum_steps
        # EMA
        ema_enabled: bool = False,
        ema_decay: float = 0.9999,
        ema_update_every: int = 1,
        ema_update_after_step: Union[int, str] = "auto",
        ema_update_after_epoch: Optional[int] = None,
        ema_min_update_after_step: int = 500,
        ema_eval_with_ema: bool = True,
    ):
        self.model    = model.to(device)
        self.loss_fn  = loss_fn
        self.device   = torch.device(device)
        self.use_amp  = self.device.type == "cuda" and torch.cuda.is_available()
        # bfloat16: Ampere+(A100/H100)에서 float16보다 안정적이고 빠름
        _bf16_ok = (
            self.use_amp
            and torch.cuda.is_bf16_supported()
        )
        self.amp_dtype = torch.bfloat16 if _bf16_ok else torch.float16
        self.max_epochs = max_epochs
        self.grad_clip  = grad_clip
        self.early_stop_patience = early_stop_patience
        self.warmup_epochs = warmup_epochs
        self.warmup_scale  = max(warmup_scale, 1e-8)
        self._schedule_type = schedule
        self.ckpt_dir       = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_name      = ckpt_name
        self.last_name      = "last.pt"
        self.epoch_callback = epoch_callback
        self.data_fraction  = data_fraction
        self._global_step   = 0
        # C²OT
        self.c2ot_sampler   = c2ot_sampler
        self.c2ot_loss_fn   = c2ot_loss_fn
        # Gradient Accumulation
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        # GradScaler: __init__에서 한 번만 생성 → scale factor가 에폭 간 유지됨
        # bfloat16은 스케일링 불필요 (언더플로우 문제 없음)
        self.scaler = GradScaler("cuda", enabled=(self.use_amp and self.amp_dtype == torch.float16))

        # Optimizer
        if optimizer == "adamw":
            self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
        elif optimizer == "sgd":
            from torch.optim import SGD
            self.optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                 momentum=momentum, nesterov=nesterov)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer!r}")

        # LR scheduler (cosine: step per epoch / plateau: step on val_loss)
        if schedule == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min,
            )
        elif schedule == "cosine_warmup":
            eta_min_ratio = eta_min / lr
            warmup = warmup_epochs
            T = T_max
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                t = (epoch - warmup) / max(T - warmup, 1)
                t = min(t, 1.0)
                return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * t))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        elif schedule == "cosine_restarts":
            self.scheduler = CosineAnnealingWarmRestarts(
                self.optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min,
            )
        elif schedule == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min",
                patience=plateau_patience, factor=plateau_factor,
            )
        else:
            raise ValueError(f"Unknown schedule: {schedule!r}")

        self._history = {"train_loss": [], "val_loss": [], "lr": []}
        self._best_val = math.inf
        self._best_epoch = -1
        self._no_improve = 0
        self._base_lr = lr
        self._last_grad_norm = 0.0
        self._last_val_source = "raw"

        # EMA
        self.ema_enabled = bool(ema_enabled)
        self.ema_decay = float(ema_decay)
        self.ema_update_every = int(ema_update_every)
        self.ema_update_after_step_cfg = ema_update_after_step
        self.ema_update_after_epoch = (
            None if ema_update_after_epoch is None else int(ema_update_after_epoch)
        )
        self.ema_min_update_after_step = int(ema_min_update_after_step)
        self.ema_eval_with_ema = bool(ema_eval_with_ema)
        self.ema_update_after_step = 0
        self.ema_updates = 0
        self.ema_model: Optional[nn.Module] = None

        if self.ema_enabled:
            if not (0.0 < self.ema_decay < 1.0):
                raise ValueError(f"ema_decay must be in (0, 1), got {self.ema_decay!r}")
            if self.ema_update_every <= 0:
                raise ValueError(f"ema_update_every must be > 0, got {self.ema_update_every!r}")
            if self.ema_min_update_after_step < 0:
                raise ValueError(
                    f"ema_min_update_after_step must be >= 0, got {self.ema_min_update_after_step!r}"
                )
            if self.ema_update_after_epoch is not None and self.ema_update_after_epoch < 0:
                raise ValueError(
                    "ema_update_after_epoch must be >= 0, "
                    f"got {self.ema_update_after_epoch!r}"
                )
            self._reset_ema_from_model()

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _warmup_step(self, epoch: int):
        if epoch < self.warmup_epochs:
            self._set_lr(self._base_lr * (epoch + 1) / (self.warmup_epochs * self.warmup_scale))

    def _warmup_lr_for_epoch(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self._base_lr
        if epoch < self.warmup_epochs:
            return self._base_lr * (epoch + 1) / (self.warmup_epochs * self.warmup_scale)
        return self._base_lr

    # ── EMA ───────────────────────────────────────────────────────────────────

    def _resolve_ema_update_after_step(self, steps_per_epoch: int) -> int:
        if self.ema_update_after_epoch is not None:
            return self.ema_update_after_epoch * steps_per_epoch
        cfg = self.ema_update_after_step_cfg
        if isinstance(cfg, str):
            mode = cfg.strip().lower()
            if mode in {"auto", "warmup", "warmup_auto"}:
                return max(self.ema_min_update_after_step, self.warmup_epochs * steps_per_epoch)
            if mode in {"0", "none"}:
                return 0
            raise ValueError(
                f"Unknown ema_update_after_step mode: {cfg!r}. "
                "Use integer or one of: auto / warmup / none"
            )
        value = int(cfg)
        if value < 0:
            raise ValueError(f"ema_update_after_step must be >= 0, got {value!r}")
        return value

    def _reset_ema_from_model(self) -> None:
        self.ema_model = deepcopy(self.model).to(self.device)
        self.ema_model.eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def _ema_update(self) -> None:
        if not self.ema_enabled or self.ema_model is None:
            return
        one_minus_decay = 1.0 - self.ema_decay
        for ema_p, p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.mul_(self.ema_decay).add_(p.detach(), alpha=one_minus_decay)
        # Keep buffers (if any) in sync for robust behavior across model variants.
        for ema_b, b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(b)
        self.ema_updates += 1

    def _current_eval_model(self) -> nn.Module:
        if self.ema_enabled and self.ema_eval_with_ema and self.ema_model is not None:
            return self.ema_model
        return self.model

    # ── Train / Val ───────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        accum = self.grad_accum_steps
        scaler = self.scaler
        is_warmup = epoch < self.warmup_epochs
        desc = "  [W]train" if is_warmup else "     train"
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not sys.stdout.isatty())

        self.optimizer.zero_grad()
        t_batch = time.time()
        for step_idx, (maps, params) in enumerate(pbar):
            maps   = maps.to(self.device, memory_format=torch.channels_last).float()
            params = params.to(self.device).float()

            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss = self.loss_fn(self.model, maps, params) / accum

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is {loss.item()*accum:.4f} at epoch {epoch+1}. Stopping training.")

            scaler.scale(loss).backward()

            if (step_idx + 1) % accum == 0:
                if self.grad_clip > 0:
                    scaler.unscale_(self.optimizer)
                    self._last_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self._global_step += 1

                if self.ema_enabled and self.ema_model is not None:
                    if (
                        self._global_step >= self.ema_update_after_step
                        and (self._global_step % self.ema_update_every == 0)
                    ):
                        self._ema_update()

            total += loss.item() * accum * maps.size(0)
            n     += maps.size(0)
            batch_sec = time.time() - t_batch
            t_batch = time.time()
            pbar.set_postfix(loss=f"{total/n:.4f}", batch=f"{batch_sec:.1f}s")
        return total / n

    def _train_epoch_c2ot(self, pairs: list, batch_size: int, epoch: int) -> float:
        """
        C²OT 전용 train epoch.
        OT로 미리 매칭된 pairs 리스트를 배치 단위로 순회.
        AMP / grad_clip / EMA / progress bar 모두 _train_epoch와 동일.
        """
        self.model.train()
        random.shuffle(pairs)

        total, n  = 0.0, 0
        accum     = self.grad_accum_steps
        scaler    = self.scaler
        is_warmup = epoch < self.warmup_epochs
        desc      = "  [W] c2ot" if is_warmup else "      c2ot"

        use_prefetch = self.device.type == "cuda"
        if use_prefetch:
            loader = _C2OTPrefetcher(pairs, batch_size, self.device)
        else:
            loader = None
        n_steps = len(pairs) // batch_size
        pbar    = tqdm(range(n_steps), desc=desc, leave=False,
                       dynamic_ncols=True, disable=not sys.stdout.isatty())

        self.optimizer.zero_grad()
        t_batch  = time.time()
        prefetch_iter = iter(loader) if use_prefetch else None
        for step_idx in pbar:
            if use_prefetch:
                x_data, x_noise, theta = next(prefetch_iter)
                b = x_data.size(0)
            else:
                batch   = pairs[step_idx * batch_size : (step_idx + 1) * batch_size]
                x_data  = torch.stack([p["x_data"]  for p in batch]).to(self.device, memory_format=torch.channels_last).float()
                x_noise = torch.stack([p["x_noise"] for p in batch]).to(self.device, memory_format=torch.channels_last).float()
                theta   = torch.stack([p["theta"]   for p in batch]).to(self.device).float()
                b       = len(batch)

            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss = self.c2ot_loss_fn(self.model, x_data, x_noise, theta) / accum

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"[C²OT] Loss is {loss.item()*accum:.4f} at epoch {epoch+1}. Stopping training."
                )

            scaler.scale(loss).backward()

            if (step_idx + 1) % accum == 0:
                if self.grad_clip > 0:
                    scaler.unscale_(self.optimizer)
                    self._last_grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    ).item()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad()
                self._global_step += 1

                if self.ema_enabled and self.ema_model is not None:
                    if (
                        self._global_step >= self.ema_update_after_step
                        and (self._global_step % self.ema_update_every == 0)
                    ):
                        self._ema_update()

            total += loss.item() * accum * b
            n     += b
            batch_sec = time.time() - t_batch
            t_batch = time.time()
            pbar.set_postfix(loss=f"{total/n:.4f}", batch=f"{batch_sec:.1f}s")

        return total / n if n > 0 else 0.0

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, eval_model: Optional[nn.Module] = None) -> float:
        model_for_eval = eval_model if eval_model is not None else self.model
        model_for_eval.eval()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc="    val", leave=False, dynamic_ncols=True, disable=not sys.stdout.isatty())
        for maps, params in pbar:
            maps   = maps.to(self.device, memory_format=torch.channels_last).float()
            params = params.to(self.device).float()
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                loss   = self.loss_fn(model_for_eval, maps, params)
            total += loss.item() * maps.size(0)
            n     += maps.size(0)
            pbar.set_postfix(loss=f"{total/n:.4f}")
        return total / n

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save(self, epoch: int, val_loss: float, name: Optional[str] = None):
        if self.ckpt_dir is None:
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch":     epoch,
            "model":     self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "val_loss":  val_loss,
            "history":   self._history,
            "best_val":  self._best_val,
            "best_epoch": self._best_epoch,
            "no_improve": self._no_improve,
            "schedule_type": self._schedule_type,
            "global_step": self._global_step,
            "val_loss_source": self._last_val_source,
            "ema_enabled": self.ema_enabled,
            "ema_decay": self.ema_decay if self.ema_enabled else None,
            "ema_update_every": self.ema_update_every if self.ema_enabled else None,
            "ema_update_after_step": self.ema_update_after_step if self.ema_enabled else None,
            "ema_updates": self.ema_updates if self.ema_enabled else None,
        }
        if self.ema_enabled and self.ema_model is not None:
            payload["model_ema"] = self.ema_model.state_dict()
        payload["scaler"] = self.scaler.state_dict()
        torch.save(payload, self.ckpt_dir / (name or self.ckpt_name))

    def load(self, path: Optional[Path] = None) -> int:
        """체크포인트 로드 후 시작 epoch 반환."""
        p = Path(path) if path is not None else None
        if p is None and self.ckpt_dir is not None:
            last_path = self.ckpt_dir / self.last_name
            best_path = self.ckpt_dir / self.ckpt_name
            p = last_path if last_path.exists() else best_path
        if p is None or not Path(p).exists():
            return 0
        ck = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        self.optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler") is not None:
            self.scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck["epoch"]) + 1

        if ck.get("scheduler") is not None:
            self.scheduler.load_state_dict(ck["scheduler"])
            scheduler_msg = "scheduler=restored"
        elif self._schedule_type != "plateau":
            # Older checkpoints do not have scheduler state. For non-plateau
            # schedulers we can still recover the closed-form state at epoch N.
            self.scheduler.step(start_epoch)
            scheduler_msg = f"scheduler=fast-forwarded({start_epoch})"
        else:
            scheduler_msg = "scheduler=fresh(no state)"

        if ck.get("history") is not None:
            self._history = ck["history"]

        if ck.get("best_val") is not None:
            self._best_val = float(ck["best_val"])
        elif self.ckpt_dir is not None:
            best_path = self.ckpt_dir / self.ckpt_name
            if best_path.exists() and best_path.resolve() != p.resolve():
                best_ck = torch.load(best_path, map_location="cpu")
                self._best_val = float(best_ck.get("best_val", best_ck.get("val_loss", math.inf)))
            else:
                self._best_val = float(ck["val_loss"])
        else:
            self._best_val = float(ck["val_loss"])

        if ck.get("best_epoch") is not None:
            self._best_epoch = int(ck["best_epoch"])
        elif self.ckpt_dir is not None:
            best_path = self.ckpt_dir / self.ckpt_name
            if best_path.exists() and best_path.resolve() != p.resolve():
                best_ck = torch.load(best_path, map_location="cpu")
                self._best_epoch = int(best_ck.get("best_epoch", int(best_ck.get("epoch", -1)) + 1))
            else:
                self._best_epoch = int(ck["epoch"]) + 1
        else:
            self._best_epoch = int(ck["epoch"]) + 1

        if ck.get("no_improve") is not None:
            self._no_improve = int(ck["no_improve"])
        elif self._best_epoch > 0:
            self._no_improve = max(0, start_epoch - self._best_epoch)
        else:
            self._no_improve = 0

        self._global_step = int(ck.get("global_step", 0))
        self._last_val_source = str(ck.get("val_loss_source", "raw"))

        ema_msg = ""
        if self.ema_enabled and self.ema_model is not None:
            ck_ema = ck.get("model_ema")
            if ck_ema is not None:
                self.ema_model.load_state_dict(ck_ema)
                self.ema_updates = int(ck.get("ema_updates", 0))
                ema_msg = "  ema=restored"
            else:
                self._reset_ema_from_model()
                self.ema_updates = 0
                ema_msg = "  ema=fallback(raw->ema copy)"
            self.ema_model.eval()

        print(
            f"Loaded: epoch={ck['epoch']}  next_epoch={start_epoch}  "
            f"val_loss={ck['val_loss']:.5f}  best={self._best_val:.5f}@ep{self._best_epoch}  "
            f"{scheduler_msg}{ema_msg}"
        )
        return start_epoch

    def load_weights_only(self, path: Union[str, Path]) -> None:
        """모델 가중치만 로드 (optimizer state / epoch 은 유지)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ck = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        ema_msg = ""
        if self.ema_enabled and self.ema_model is not None:
            ck_ema = ck.get("model_ema")
            if ck_ema is not None:
                self.ema_model.load_state_dict(ck_ema)
                self.ema_updates = int(ck.get("ema_updates", 0))
                ema_msg = "  ema=loaded"
            else:
                self._reset_ema_from_model()
                self.ema_updates = 0
                ema_msg = "  ema=fallback(raw->ema copy)"
            self.ema_model.eval()
        self._global_step = 0
        print(
            f"Weights loaded: epoch={ck['epoch']}  val_loss={ck['val_loss']:.5f}  "
            f"(optimizer reset){ema_msg}"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        start_epoch:  int = 0,
    ) -> dict:
        n_train = len(train_loader.dataset)
        n_val   = len(val_loader.dataset)
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6

        # C²OT 모드에서는 배치 크기 = DataLoader batch_size, steps = n_train // batch_size
        c2ot_mode       = self.c2ot_sampler is not None and self.c2ot_loss_fn is not None
        c2ot_batch_size = train_loader.batch_size or 16
        steps_per_epoch = (n_train // c2ot_batch_size) if c2ot_mode else len(train_loader)

        gpu_name = torch.cuda.get_device_name(self.device) if self.use_amp else "cpu"
        gpu_mem  = torch.cuda.get_device_properties(self.device).total_memory / 1e9 if self.use_amp else 0

        frac_str = f"  ({self.data_fraction*100:.0f}% of full)" if self.data_fraction < 1.0 else ""
        warmup_start_lr = self._warmup_lr_for_epoch(0) if self.warmup_epochs > 0 else self._base_lr
        if self.ema_enabled:
            self.ema_update_after_step = self._resolve_ema_update_after_step(steps_per_epoch)
        else:
            self.ema_update_after_step = 0

        val_source = "ema" if (self.ema_enabled and self.ema_eval_with_ema and self.ema_model is not None) else "raw"
        if val_source == "ema":
            self._last_val_source = "ema"

        print("=" * 60)
        print("  GENESIS Training Start")
        print("=" * 60)
        print(f"  Device     : {self.device} ({gpu_name})" + (f"  [{gpu_mem:.1f}GB]" if self.use_amp else ""))
        print(f"  Model      : {self.model.__class__.__name__}  ({n_params:.1f}M params)")
        c2ot_tag = f"  [C²OT batch={c2ot_batch_size}]" if c2ot_mode else ""
        print(f"  Data       : train={n_train}{frac_str}  val={n_val}  batches/ep={steps_per_epoch}{c2ot_tag}")
        print(f"  Epochs     : {start_epoch} → {self.max_epochs}  (warmup={self.warmup_epochs})")
        print(f"  Optimizer  : {self.optimizer.__class__.__name__}  lr={self._base_lr:.1e}  wd={self.optimizer.param_groups[0].get('weight_decay', 0):.1e}")
        eff_batch = (c2ot_batch_size if c2ot_mode else train_loader.batch_size) * self.grad_accum_steps
        print(f"  Schedule   : {self._schedule_type}  grad_clip={self.grad_clip}  grad_accum={self.grad_accum_steps}  eff_batch={eff_batch}")
        if self.warmup_epochs > 0:
            print(
                f"  Warmup     : enabled  epochs=1..{self.warmup_epochs}  "
                f"lr={warmup_start_lr:.1e}→{self._base_lr:.1e}"
            )
        else:
            print("  Warmup     : disabled")
        if self.ema_enabled:
            epoch_cfg = (
                "none" if self.ema_update_after_epoch is None else str(self.ema_update_after_epoch)
            )
            print(
                f"  EMA        : enabled  decay={self.ema_decay:.6f}  update_every={self.ema_update_every}  "
                f"update_after_step={self.ema_update_after_step}  update_after_epoch={epoch_cfg}  "
                f"val_source={val_source}"
            )
            if self._schedule_type == "plateau" and val_source == "ema":
                print("  NOTE       : plateau scheduler uses EMA val_loss (scheduler.step(val_loss)).")
        else:
            print("  EMA        : disabled")
        print(f"  Early stop : patience={self.early_stop_patience}")
        print(f"  Checkpoint : {self.ckpt_dir / self.ckpt_name if self.ckpt_dir else 'none'}")
        print("=" * 60)
        print(self.model)
        print("=" * 60)

        epoch_times = []

        _compiled = getattr(self.model, "_is_compiled", False) or (
            type(self.model).__name__ == "OptimizedModule"
        )

        for epoch in range(start_epoch, self.max_epochs):
            t0 = time.time()

            if self._schedule_type not in ("cosine_warmup",):
                self._warmup_step(epoch)
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])

            if self.use_amp:
                torch.cuda.reset_peak_memory_stats(self.device)

            if _compiled and epoch == start_epoch:
                print("[train] 첫 에폭: torch.compile 커널 컴파일 중 (배치마다 처음엔 느림, 이후 정상)")

            if c2ot_mode:
                pair_t0 = time.time()
                print(
                    f"[train] ep{epoch+1:04d}: C2OT pair matching start "
                    f"(dataset={len(train_loader.dataset)})"
                )
                pairs = self.c2ot_sampler.compute_pairs(train_loader.dataset)
                print(
                    f"[train] ep{epoch+1:04d}: C2OT pair matching done "
                    f"(pairs={len(pairs)}, {time.time() - pair_t0:.1f}s)"
                )
                train_loss = self._train_epoch_c2ot(pairs, c2ot_batch_size, epoch)
            else:
                train_loss = self._train_epoch(train_loader, epoch)
            eval_model = self._current_eval_model()
            self._last_val_source = "ema" if eval_model is self.ema_model else "raw"
            val_loss   = self._val_epoch(val_loader, eval_model=eval_model)

            train_val_elapsed = time.time() - t0

            if self.use_amp:
                mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9

            self._history["train_loss"].append(train_loss)
            self._history["val_loss"].append(val_loss)
            self._history["lr"].append(epoch_lr)

            # Scheduler step
            if epoch >= self.warmup_epochs or self._schedule_type in ("cosine_warmup", "cosine_restarts"):
                if self._schedule_type == "cosine":
                    self.scheduler.step()
                elif self._schedule_type in ("cosine_warmup", "cosine_restarts"):
                    self.scheduler.step()
                elif self._schedule_type == "plateau":
                    # NOTE: val_loss source(raw vs ema)는 self._last_val_source로 추적된다.
                    self.scheduler.step(val_loss)

            # Best checkpoint
            improved = val_loss < self._best_val
            if improved:
                self._best_val   = val_loss
                self._best_epoch = epoch + 1
                self._no_improve = 0
                self._save(epoch, val_loss)
            else:
                self._no_improve += 1

            self._save(epoch, val_loss, name=self.last_name)

            is_warmup    = epoch < self.warmup_epochs
            warmup_tag   = "[W]" if is_warmup else "   "
            warmup_info  = (
                f"  warmup={epoch+1}/{self.warmup_epochs}"
                if is_warmup and self.warmup_epochs > 0 else ""
            )
            mem_str      = f"  mem={mem_gb:.1f}GB" if self.use_amp else ""
            patience_str = f"patience={self._no_improve}/{self.early_stop_patience}"
            best_str     = f"best={self._best_val:.5f}@ep{self._best_epoch}"
            best_mark    = " *" if improved else ""
            callback_elapsed = 0.0

            # Epoch callback (시각화 등)
            if self.epoch_callback is not None:
                cb_t0 = time.time()
                callback_model = eval_model
                try:
                    self.epoch_callback(epoch, callback_model, self._history, improved)
                except TypeError:
                    # Backward compatibility for 3-argument callbacks
                    self.epoch_callback(epoch, callback_model, self._history)
                if self.ema_enabled and self.ema_model is not None:
                    self.ema_model.eval()
                torch.cuda.empty_cache()   # viz 샘플링 후 caching allocator 정리
                callback_elapsed = time.time() - cb_t0

            total_elapsed = time.time() - t0
            epoch_times.append(total_elapsed)

            # ETA
            avg_epoch_sec = sum(epoch_times) / len(epoch_times)
            remaining_epochs = self.max_epochs - (epoch + 1)
            eta_sec = avg_epoch_sec * remaining_epochs
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

            timing_parts = [f"train+val={train_val_elapsed:.1f}s"]
            if self.epoch_callback is not None:
                timing_parts.append(f"cb={callback_elapsed:.1f}s")
            timing_parts.append(f"total={total_elapsed:.1f}s/ep")
            timing_str = "  ".join(timing_parts)

            print(
                f"{warmup_tag}[{epoch+1:04d}/{self.max_epochs}] "
                f"train={train_loss:.5f}  val={val_loss:.5f}  lr={epoch_lr:.2e}"
                f"{warmup_info}"
                f"  gnorm={self._last_grad_norm:.2f}{mem_str}"
                f"  {timing_str}  eta={eta_str}"
                f"  {patience_str}  {best_str}{best_mark}"
            )

            # Early stopping
            if self._no_improve >= self.early_stop_patience:
                print(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.early_stop_patience} epochs)"
                )
                break

        total_sec = sum(epoch_times)
        print(
            f"[trainer] done.  best_val={self._best_val:.5f}@ep{self._best_epoch}"
            f"  total={time.strftime('%H:%M:%S', time.gmtime(total_sec))}"
        )

        return self._history
