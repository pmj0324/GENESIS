import h5py
import numpy as np

# HDF5 파일 경로 설정
path = "../../GENESIS-data/22644_0921_time_shift.h5"

# 파일 열기
with h5py.File(path, "r") as f:
    x = f["input"][:]  # shape (178056, 2, 5160)

# nonzero 개수 계산
# axis=(1,2) → 각 이벤트(178056개)에 대해 2×5160 값 중 nonzero 카운트
nonzero_counts = np.count_nonzero(x, axis=(1,2))

# nonzero 개수 가장 적은 15개 인덱스
idx_sorted = np.argsort(nonzero_counts)[-15:]

print("Top 15 samples with smallest nonzero count:")
for i in idx_sorted:
    print(f"Index {i}: nonzero={nonzero_counts[i]}")
