"""
GENESIS - Trainer

AdamW + Cosine / Plateau schedule + warmup + early stopping.
Diffusion (.loss) 과 Flow Matching (.loss) 모두 지원.
"""

import math
import sys
import time
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
    ):
        self.model    = model.to(device)
        self.loss_fn  = loss_fn
        self.device   = torch.device(device)
        self.use_amp  = self.device.type == "cuda" and torch.cuda.is_available()
        self.max_epochs = max_epochs
        self.grad_clip  = grad_clip
        self.early_stop_patience = early_stop_patience
        self.warmup_epochs = warmup_epochs
        self._schedule_type = schedule
        self.ckpt_dir       = Path(ckpt_dir) if ckpt_dir else None
        self.ckpt_name      = ckpt_name
        self.last_name      = "last.pt"
        self.epoch_callback = epoch_callback
        self.data_fraction  = data_fraction

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

    # ── Warmup ────────────────────────────────────────────────────────────────

    def _set_lr(self, lr: float):
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _warmup_step(self, epoch: int):
        if epoch < self.warmup_epochs:
            self._set_lr(self._base_lr * (epoch + 1) / self.warmup_epochs)

    def _warmup_lr_for_epoch(self, epoch: int) -> float:
        if self.warmup_epochs <= 0:
            return self._base_lr
        if epoch < self.warmup_epochs:
            return self._base_lr * (epoch + 1) / self.warmup_epochs
        return self._base_lr

    # ── Train / Val ───────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total, n = 0.0, 0
        scaler = GradScaler("cuda", enabled=self.use_amp)
        is_warmup = epoch < self.warmup_epochs
        desc = "  [W]train" if is_warmup else "     train"
        pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, disable=not sys.stdout.isatty())
        for maps, params in pbar:
            # Normalize/normalization pipeline changes can accidentally produce float64.
            # Force float32 to match model weights (conv bias float32).
            maps   = maps.to(self.device).float()    # [B, 3, 256, 256]
            params = params.to(self.device).float()  # [B, 6]

            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.use_amp):
                loss = self.loss_fn(self.model, maps, params)

            if not torch.isfinite(loss):
                raise RuntimeError(f"Loss is {loss.item():.4f} at epoch {epoch+1}. Stopping training.")

            scaler.scale(loss).backward()
            if self.grad_clip > 0:
                # clip_grad_norm_ expects unscaled gradients
                scaler.unscale_(self.optimizer)
                self._last_grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip).item()
            scaler.step(self.optimizer)
            scaler.update()

            total += loss.item() * maps.size(0)
            n     += maps.size(0)
            pbar.set_postfix(loss=f"{total/n:.4f}")
        return total / n

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total, n = 0.0, 0
        pbar = tqdm(loader, desc="    val", leave=False, dynamic_ncols=True, disable=not sys.stdout.isatty())
        for maps, params in pbar:
            maps   = maps.to(self.device).float()
            params = params.to(self.device).float()
            with autocast("cuda", enabled=self.use_amp):
                loss   = self.loss_fn(self.model, maps, params)
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
        }
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

        print(
            f"Loaded: epoch={ck['epoch']}  next_epoch={start_epoch}  "
            f"val_loss={ck['val_loss']:.5f}  best={self._best_val:.5f}@ep{self._best_epoch}  "
            f"{scheduler_msg}"
        )
        return start_epoch

    def load_weights_only(self, path: Union[str, Path]) -> None:
        """모델 가중치만 로드 (optimizer state / epoch 은 유지)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        ck = torch.load(p, map_location=self.device)
        self.model.load_state_dict(ck["model"])
        print(f"Weights loaded: epoch={ck['epoch']}  val_loss={ck['val_loss']:.5f}  (optimizer reset)")

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

        gpu_name = torch.cuda.get_device_name(self.device) if self.use_amp else "cpu"
        gpu_mem  = torch.cuda.get_device_properties(self.device).total_memory / 1e9 if self.use_amp else 0

        frac_str = f"  ({self.data_fraction*100:.0f}% of full)" if self.data_fraction < 1.0 else ""
        warmup_start_lr = self._warmup_lr_for_epoch(0) if self.warmup_epochs > 0 else self._base_lr
        print("=" * 60)
        print("  GENESIS Training Start")
        print("=" * 60)
        print(f"  Device     : {self.device} ({gpu_name})" + (f"  [{gpu_mem:.1f}GB]" if self.use_amp else ""))
        print(f"  Model      : {self.model.__class__.__name__}  ({n_params:.1f}M params)")
        print(f"  Data       : train={n_train}{frac_str}  val={n_val}  batches/ep={len(train_loader)}")
        print(f"  Epochs     : {start_epoch} → {self.max_epochs}  (warmup={self.warmup_epochs})")
        print(f"  Optimizer  : {self.optimizer.__class__.__name__}  lr={self._base_lr:.1e}  wd={self.optimizer.param_groups[0].get('weight_decay', 0):.1e}")
        print(f"  Schedule   : {self._schedule_type}  grad_clip={self.grad_clip}")
        if self.warmup_epochs > 0:
            print(
                f"  Warmup     : enabled  epochs=1..{self.warmup_epochs}  "
                f"lr={warmup_start_lr:.1e}→{self._base_lr:.1e}"
            )
        else:
            print("  Warmup     : disabled")
        print(f"  Early stop : patience={self.early_stop_patience}")
        print(f"  Checkpoint : {self.ckpt_dir / self.ckpt_name if self.ckpt_dir else 'none'}")
        print("=" * 60)
        print(self.model)
        print("=" * 60)

        epoch_times = []

        for epoch in range(start_epoch, self.max_epochs):
            t0 = time.time()

            if self._schedule_type not in ("cosine_warmup",):
                self._warmup_step(epoch)
            epoch_lr = float(self.optimizer.param_groups[0]["lr"])

            if self.use_amp:
                torch.cuda.reset_peak_memory_stats(self.device)

            train_loss = self._train_epoch(train_loader, epoch)
            val_loss   = self._val_epoch(val_loader)

            elapsed = time.time() - t0

            if self.use_amp:
                mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9
            epoch_times.append(elapsed)

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

            # ETA
            avg_epoch_sec = sum(epoch_times) / len(epoch_times)
            remaining_epochs = self.max_epochs - (epoch + 1)
            eta_sec = avg_epoch_sec * remaining_epochs
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_sec))

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

            print(
                f"{warmup_tag}[{epoch+1:04d}/{self.max_epochs}] "
                f"train={train_loss:.5f}  val={val_loss:.5f}  lr={epoch_lr:.2e}"
                f"{warmup_info}"
                f"  gnorm={self._last_grad_norm:.2f}{mem_str}"
                f"  {elapsed:.1f}s/ep  eta={eta_str}"
                f"  {patience_str}  {best_str}{best_mark}"
            )

            # Epoch callback (시각화 등)
            if self.epoch_callback is not None:
                self.epoch_callback(epoch, self.model, self._history)
                torch.cuda.empty_cache()   # viz 샘플링 후 caching allocator 정리

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
