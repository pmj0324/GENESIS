import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm
import numpy as np

def get_image_loaders(
    h5_path,
    batch_size=64,
    seed=42,
    div0=1.0, div1=1.0, shift0=0.0, shift1=0.0,
    scale_first_time="none",
    num_workers=8, prefetch_factor=4
):
    dataset = PMTImageDataset(h5_path, div0, div1, shift0, shift1, scale_first_time)
    N = len(dataset)
    n_val = int(N * 0.1)
    n_test = int(N * 0.1)
    n_train = N - n_val - n_test
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch_factor
    )
    return train_loader, val_loader, test_loader

def train_conditional_ddpm_image(
    h5_path,
    num_epochs=100,
    batch_size=128,
    T=1000,
    patience=10,
    save_path="best_image_ddpm.pth",
    div0=200.0,
    div1=100000.0,
    shift0=0.0,
    shift1=0.0,
    scale_first_time="none"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_image_loaders(
        h5_path, batch_size, 42, div0, div1, shift0, shift1, scale_first_time
    )
    beta = make_beta_schedule(T)
    _, alpha_bar = get_alpha_bar(beta)
    model = SimpleCondUNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    best_val = float('inf')
    patience_cnt = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_iter = tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}/{num_epochs}", ncols=110)
        for x, c in train_iter:
            x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
            bsz = x.size(0)
            t = torch.randint(0, T, (bsz,), device=device)
            with autocast():
                x_noised, noise = q_sample(x, t, alpha_bar)
                pred = model(x_noised, t, c)
                loss = criterion(pred, noise)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * bsz
            train_iter.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        val_iter = tqdm(val_loader, desc="[Val]", ncols=110, leave=False)
        with torch.inference_mode():
            for x, c in val_iter:
                x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
                bsz = x.size(0)
                t = torch.randint(0, T, (bsz,), device=device)
                x_noised, noise = q_sample(x, t, alpha_bar)
                pred = model(x_noised, t, c)
                loss = criterion(pred, noise)
                val_loss += loss.item() * bsz
                val_iter.set_postfix(loss=loss.item())
        val_loss /= len(val_loader.dataset)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print("Early stopping!")
                break

    # 테스트
    print("Evaluating on test set...")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    test_loss = 0
    test_iter = tqdm(test_loader, desc="[Test]", ncols=110)
    with torch.inference_mode():
        for x, c in test_iter:
            x, c = x.to(device, non_blocking=True), c.to(device, non_blocking=True)
            bsz = x.size(0)
            t = torch.randint(0, T, (bsz,), device=device)
            x_noised, noise = q_sample(x, t, alpha_bar)
            pred = model(x_noised, t, c)
            loss = criterion(pred, noise)
            test_loss += loss.item() * bsz
            test_iter.set_postfix(loss=loss.item())
    test_loss /= len(test_loader.dataset)
    print(f"Test loss: {test_loss:.6f}")
