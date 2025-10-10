#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
h5_hist.py - Advanced Histogram Plotter for IceCube HDF5 Data
================================================================

HDF5 파일의 input 데이터셋 (shape: N, 2, 5160)에서
채널 0=charge(NPE), 채널 1=time 의 고급 히스토그램을 생성합니다.

Features:
- 로그 스케일 (X축, Y축)
- 통계선 표시 (평균, 중앙값, 표준편차)
- 퍼센타일선 표시 (P10, P25, P75, P90)
- 통계 정보 박스
- 스트리밍 처리 (대용량 파일 지원)

사용 예시:
================================================================

1. 기본 사용 (모든 통계선 표시):
   python utils/h5_hist.py -p /path/to/data.h5

2. 로그 스케일 적용:
   python utils/h5_hist.py -p /path/to/data.h5 --logy --logx

3. 통계선 숨김 (히스토그램만):
   python utils/h5_hist.py -p /path/to/data.h5 --no-stats --no-percentiles

4. 퍼센타일만 표시:
   python utils/h5_hist.py -p /path/to/data.h5 --no-stats

5. 큰 그림 크기 + 자세한 범위:
   python utils/h5_hist.py -p /path/to/data.h5 --figsize 16 10 \
     --range-charge 0 100 --range-time 0 50000

6. 고해상도 + 많은 빈:
   python utils/h5_hist.py -p /path/to/data.h5 --bins 500 --chunk 2048

7. 0 제외 히스토그램:
   python utils/h5_hist.py -p /path/to/data.h5 --exclude-zero

8. 모든 값과 0 제외 값 모두 플롯:
   python utils/h5_hist.py -p /path/to/data.h5 --plot-both

9. Time 임계값 적용 (1000ns 이상만):
   python utils/h5_hist.py -p /path/to/data.h5 --min-time 1000

10. Time 값에 log10 변환 적용:
    python utils/h5_hist.py -p /path/to/data.h5 --log-time log10

11. Time 값에 ln 변환 적용하고 0 제외:
    python utils/h5_hist.py -p /path/to/data.h5 --log-time ln --exclude-zero

12. 완전 커스텀 설정:
   python utils/h5_hist.py -p /path/to/data.h5 \
     --bins 300 --logy --logx --figsize 14 8 \
     --range-charge 0 50 --range-time 0 20000 \
     --out custom_hist --pclip 1 99

CLI Options:
================================================================
-p, --path          : HDF5 파일 경로 (필수)
--bins              : 히스토그램 빈 개수 (기본: 200)
--chunk             : 스트리밍 배치 크기 (기본: 1024)
--range-charge      : Charge 수동 범위 (예: 0 50)
--range-time        : Time 수동 범위 (예: 0 20000)
--out               : 출력 파일 접두사 (기본: hist_input)
--logy              : Y축 로그 스케일
--logx              : X축 로그 스케일
--log-time          : Time 값 로그 변환 (log10, ln) - 기본값: log10
--no-stats          : 통계선 숨김 (평균, 중앙값, 표준편차)
--no-percentiles    : 퍼센타일선 숨김 (P10, P90, P25, P75)
--figsize           : 그림 크기 (예: 12 8)
--exclude-zero      : 0 값을 제외한 히스토그램만 생성
--plot-both         : 모든 값과 0 제외 값 모두 플롯
--min-time          : Time 최소 임계값 (ns) - 이 값 이상의 time만 플롯
--pclip             : 자동 범위 퍼센타일 (기본: 0.5 99.5)

출력:
================================================================
- {out_prefix}_charge.png : Charge 분포 히스토그램
- {out_prefix}_time.png   : Time 분포 히스토그램
- {out_prefix}_charge_nonzero.png : Charge 분포 히스토그램 (0 제외)
- {out_prefix}_time_nonzero.png   : Time 분포 히스토그램 (0 제외)
- {out_prefix}_time_min{threshold}.png : Time 분포 히스토그램 (임계값 적용)
- 콘솔에 상세 통계 정보 출력 (0 개수, 비율, 임계값 정보 포함)

예시 출력 통계:
📊 Charge Statistics:
  Range: [0.000, 225.000]
  Mean ± Std: 0.675 ± 6.265
  Median: 0.000
  Percentiles: P10=0.000, P90=2.000

⏱️ Time Statistics:
  Range: [0.0, 135232.0]
  Mean ± Std: 624.2 ± 3110.0
  Median: 0.0
  Percentiles: P10=0.0, P90=2000.0
"""

import argparse
from typing import Optional, Tuple, Dict, Any
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import stats

def _percentile_range(
    dset, ch: int, chunk: int, sample_chunks: int, p_low: float, p_high: float
) -> Tuple[float, float]:
    """퍼센타일 기반 자동 범위 추정 (극단값에 둔감)."""
    N = dset.shape[0]
    idx = np.linspace(0, N - 1, num=min(sample_chunks * chunk, N), dtype=int)
    xs = []
    for i in range(0, len(idx), chunk):
        ids = idx[i : i + chunk]
        batch = np.asarray(dset[ids, ch, :], dtype=np.float64).ravel()
        xs.append(batch[~np.isnan(batch)])
    x = np.concatenate(xs) if xs else np.array([], dtype=np.float64)
    if x.size == 0:
        return (0.0, 1.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)

def _hist_stream(
    dset, ch: int, bins: int, v_range: Tuple[float, float], chunk: int, 
    exclude_zero: bool = False, min_threshold: Optional[float] = None,
    log_transform: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """스트리밍으로 전체 히스토그램 계산 및 통계 수집."""
    edges = np.linspace(v_range[0], v_range[1], bins + 1)
    counts = np.zeros(bins, dtype=np.int64)
    
    # 통계 수집을 위한 배열
    all_values = []
    zero_count = 0
    total_count = 0
    N = dset.shape[0]
    
    for i in range(0, N, chunk):
        sl = slice(i, min(i + chunk, N))
        x = np.asarray(dset[sl, ch, :], dtype=np.float64).ravel()
        x = x[~np.isnan(x)]
        
        # 0 카운팅
        total_count += len(x)
        zero_count += np.sum(x == 0)
        
        # 0 제외 옵션
        if exclude_zero:
            x = x[x != 0]
        
        # 최소 임계값 적용
        if min_threshold is not None:
            x = x[x >= min_threshold]
        
        # 로그 변환 적용
        if log_transform is not None and len(x) > 0:
            if log_transform == "log10":
                x = np.log10(x + 1e-10)  # 0 방지를 위한 작은 값 추가
            elif log_transform == "ln":
                x = np.log(x + 1e-10)  # 0 방지를 위한 작은 값 추가
        
        # 통계용 데이터 수집 (샘플링으로 메모리 절약)
        if len(all_values) < 100000:  # 최대 10만개 샘플
            all_values.extend(x.tolist())
        
        if len(x) > 0:
            # 범위 확인 및 조정
            if edges[0] < edges[-1]:  # 유효한 범위인지 확인
                x = np.clip(x, edges[0], edges[-1])
                c, _ = np.histogram(x, bins=edges)
                counts += c
    
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # 통계 계산
    if all_values:
        all_values = np.array(all_values)
        stats_dict = {
            'mean': np.mean(all_values),
            'std': np.std(all_values),
            'median': np.median(all_values),
            'min': np.min(all_values),
            'max': np.max(all_values),
            'p10': np.percentile(all_values, 10),
            'p90': np.percentile(all_values, 90),
            'p25': np.percentile(all_values, 25),
            'p75': np.percentile(all_values, 75),
            'count': len(all_values),
            'zero_count': zero_count,
            'total_count': total_count,
            'zero_fraction': zero_count / total_count if total_count > 0 else 0,
            'min_threshold': min_threshold
        }
    else:
        stats_dict = {
            'mean': 0, 'std': 0, 'median': 0, 'min': 0, 'max': 0,
            'p10': 0, 'p90': 0, 'p25': 0, 'p75': 0, 'count': 0,
            'zero_count': zero_count, 'total_count': total_count,
            'zero_fraction': zero_count / total_count if total_count > 0 else 0,
            'min_threshold': min_threshold
        }
    
    return centers, counts, stats_dict

def plot_hist_pair(
    h5_path: str,
    bins: int = 200,
    chunk: int = 1024,
    range_charge: Optional[Tuple[float, float]] = None,
    range_time: Optional[Tuple[float, float]] = None,
    out_prefix: str = "hist_input",
    logy: bool = False,
    logx: bool = False,
    pclip: Tuple[float, float] = (0.5, 99.5),
    show_stats: bool = True,
    show_percentiles: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    exclude_zero: bool = False,
    plot_both: bool = False,
    min_time_threshold: Optional[float] = None,
    log_time_transform: Optional[str] = None,
):
    """고급 히스토그램 플롯 함수."""
    
    with h5py.File(h5_path, "r") as f:
        if "input" not in f:
            raise KeyError("Dataset 'input' not found")
        dset = f["input"]  # (N, 2, 5160)

        # 자동 범위 추정(퍼센타일) 또는 사용자 지정 범위 사용
        if range_charge is None:
            range_charge = _percentile_range(dset, 0, chunk, sample_chunks=8,
                                             p_low=pclip[0], p_high=pclip[1])
        if range_time is None:
            range_time = _percentile_range(dset, 1, chunk, sample_chunks=8,
                                           p_low=pclip[0], p_high=pclip[1])
            
        # 임계값이 있으면 범위 조정
        if min_time_threshold is not None:
            range_time = (min_time_threshold, max(range_time[1], min_time_threshold + 1))

        # 스트리밍 히스토그램 및 통계 수집
        x_c, y_c, stats_c = _hist_stream(dset, 0, bins, range_charge, chunk, exclude_zero, None, None)
        x_t, y_t, stats_t = _hist_stream(dset, 1, bins, range_time, chunk, exclude_zero, min_time_threshold, log_time_transform)

    # Charge 히스토그램
    fig, ax = plt.subplots(figsize=figsize)
    
    # 히스토그램 플롯 (더 명확하게)
    if len(x_c) > 0 and np.sum(y_c) > 0:
        # 범위 계산
        x_min, x_max = np.min(x_c), np.max(x_c)
        if x_min < x_max:  # 유효한 범위인지 확인
            # 막대 히스토그램으로 변경
            ax.hist(x_c, bins=bins, weights=y_c, alpha=0.7, color='blue', 
                    edgecolor='darkblue', linewidth=0.5, range=(x_min, x_max))
            
            # 라인도 추가 (선택적)
            ax.plot(x_c, y_c, drawstyle="steps-mid", linewidth=2, color='darkblue', alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'Invalid data range', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=14, color='red')
    else:
        ax.text(0.5, 0.5, 'No data in this range', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, color='red')
    
    # 통계선 추가
    if show_stats:
        # 평균선
        ax.axvline(stats_c['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {stats_c["mean"]:.3f}')
        
        # 중앙값선
        ax.axvline(stats_c['median'], color='orange', linestyle=':', linewidth=2, 
                   label=f'Median: {stats_c["median"]:.3f}')
        
        # 표준편차 영역 (평균 ± 1σ)
        std_low = stats_c['mean'] - stats_c['std']
        std_high = stats_c['mean'] + stats_c['std']
        ax.axvspan(std_low, std_high, alpha=0.2, color='red', 
                   label=f'±1σ: [{std_low:.3f}, {std_high:.3f}]')
    
    # 퍼센타일선 추가
    if show_percentiles:
        ax.axvline(stats_c['p10'], color='green', linestyle='-.', alpha=0.7, 
                   label=f'P10: {stats_c["p10"]:.3f}')
        ax.axvline(stats_c['p90'], color='green', linestyle='-.', alpha=0.7, 
                   label=f'P90: {stats_c["p90"]:.3f}')
        ax.axvline(stats_c['p25'], color='purple', linestyle=':', alpha=0.7, 
                   label=f'P25: {stats_c["p25"]:.3f}')
        ax.axvline(stats_c['p75'], color='purple', linestyle=':', alpha=0.7, 
                   label=f'P75: {stats_c["p75"]:.3f}')
    
    # 축 설정
    ax.set_xlabel("Charge (NPE)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    title = "Charge Distribution (Non-zero only)" if exclude_zero else "Charge Distribution"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if logy: 
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)", fontsize=12)
    if logx: 
        ax.set_xscale("log")
        ax.set_xlabel("Charge (NPE, log scale)", fontsize=12)
    
    # X축 범위 자동 조정 (데이터에 맞게)
    if len(x_c) > 0 and np.max(y_c) > 0:
        data_min = np.min(x_c[y_c > 0])
        data_max = np.max(x_c[y_c > 0])
        margin = (data_max - data_min) * 0.05  # 5% 마진
        ax.set_xlim(max(0, data_min - margin), data_max + margin)
    
    # 범례
    if show_stats or show_percentiles:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 통계 텍스트 박스
    zero_info = f"Zeros: {stats_c['zero_count']:,} ({stats_c['zero_fraction']:.1%})" if not exclude_zero else ""
    stats_text = f"""Statistics:
Count: {stats_c['count']:,}
{zero_info}
Min: {stats_c['min']:.3f}
Max: {stats_c['max']:.3f}
Mean: {stats_c['mean']:.3f} ± {stats_c['std']:.3f}
Median: {stats_c['median']:.3f}
P10: {stats_c['p10']:.3f}
P90: {stats_c['p90']:.3f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    suffix = "_nonzero" if exclude_zero else ""
    plt.savefig(f"{out_prefix}_charge{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Time 히스토그램
    fig, ax = plt.subplots(figsize=figsize)
    
    # 히스토그램 플롯 (더 명확하게)
    if len(x_t) > 0 and np.sum(y_t) > 0:
        # 범위 계산
        x_min, x_max = np.min(x_t), np.max(x_t)
        if x_min < x_max:  # 유효한 범위인지 확인
            # 막대 히스토그램으로 변경
            ax.hist(x_t, bins=bins, weights=y_t, alpha=0.7, color='green', 
                    edgecolor='darkgreen', linewidth=0.5, range=(x_min, x_max))
            
            # 라인도 추가 (선택적)
            ax.plot(x_t, y_t, drawstyle="steps-mid", linewidth=2, color='darkgreen', alpha=0.8)
        else:
            ax.text(0.5, 0.5, 'Invalid data range', transform=ax.transAxes, 
                    ha='center', va='center', fontsize=14, color='red')
    else:
        ax.text(0.5, 0.5, 'No data in this range', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14, color='red')
    
    # 통계선 추가
    if show_stats:
        # 평균선
        ax.axvline(stats_t['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {stats_t["mean"]:.1f}')
        
        # 중앙값선
        ax.axvline(stats_t['median'], color='orange', linestyle=':', linewidth=2, 
                   label=f'Median: {stats_t["median"]:.1f}')
        
        # 표준편차 영역
        std_low = stats_t['mean'] - stats_t['std']
        std_high = stats_t['mean'] + stats_t['std']
        ax.axvspan(std_low, std_high, alpha=0.2, color='red', 
                   label=f'±1σ: [{std_low:.1f}, {std_high:.1f}]')
    
    # 퍼센타일선 추가
    if show_percentiles:
        ax.axvline(stats_t['p10'], color='blue', linestyle='-.', alpha=0.7, 
                   label=f'P10: {stats_t["p10"]:.1f}')
        ax.axvline(stats_t['p90'], color='blue', linestyle='-.', alpha=0.7, 
                   label=f'P90: {stats_t["p90"]:.1f}')
        ax.axvline(stats_t['p25'], color='purple', linestyle=':', alpha=0.7, 
                   label=f'P25: {stats_t["p25"]:.1f}')
        ax.axvline(stats_t['p75'], color='purple', linestyle=':', alpha=0.7, 
                   label=f'P75: {stats_t["p75"]:.1f}')
    
    # 축 설정
    xlabel = "Time (ns)"
    if log_time_transform == "log10":
        xlabel = "log₁₀(Time + ε) (ns)"
    elif log_time_transform == "ln":
        xlabel = "ln(Time + ε) (ns)"
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    
    title = "Time Distribution"
    if exclude_zero:
        title += " (Non-zero only)"
    if min_time_threshold is not None:
        title += f" (≥{min_time_threshold:.1f}ns)"
    if log_time_transform is not None:
        transform_name = "log₁₀" if log_time_transform == "log10" else "ln"
        title += f" [{transform_name} transformed]"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    if logy: 
        ax.set_yscale("log")
        ax.set_ylabel("Count (log scale)", fontsize=12)
    if logx: 
        ax.set_xscale("log")
        ax.set_xlabel("Time (ns, log scale)", fontsize=12)
    
    # X축 범위 자동 조정 (데이터에 맞게)
    if len(x_t) > 0 and np.max(y_t) > 0:
        data_min = np.min(x_t[y_t > 0])
        data_max = np.max(x_t[y_t > 0])
        margin = (data_max - data_min) * 0.05  # 5% 마진
        ax.set_xlim(max(0, data_min - margin), data_max + margin)
    
    # 범례
    if show_stats or show_percentiles:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 통계 텍스트 박스
    zero_info = f"Zeros: {stats_t['zero_count']:,} ({stats_t['zero_fraction']:.1%})" if not exclude_zero else ""
    threshold_info = f"Threshold: ≥{stats_t['min_threshold']:.1f}ns" if stats_t['min_threshold'] is not None else ""
    transform_info = f"Transform: {log_time_transform}" if log_time_transform is not None else ""
    
    stats_text = f"""Statistics:
Count: {stats_t['count']:,}
{zero_info}
{threshold_info}
{transform_info}
Min: {stats_t['min']:.1f}
Max: {stats_t['max']:.1f}
Mean: {stats_t['mean']:.1f} ± {stats_t['std']:.1f}
Median: {stats_t['median']:.1f}
P10: {stats_t['p10']:.1f}
P90: {stats_t['p90']:.1f}"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', fontsize=9, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    suffix = "_nonzero" if exclude_zero else ""
    if min_time_threshold is not None:
        suffix += f"_min{min_time_threshold:.0f}"
    if log_time_transform is not None:
        suffix += f"_{log_time_transform}"
    plt.savefig(f"{out_prefix}_time{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 통계 정보 출력
    zero_suffix = " (Non-zero only)" if exclude_zero else ""
    threshold_suffix = f" (≥{min_time_threshold:.1f}ns)" if min_time_threshold is not None else ""
    transform_suffix = f" [{log_time_transform} transformed]" if log_time_transform is not None else ""
    
    print(f"\n📊 Charge Statistics{zero_suffix}:")
    print(f"  Range: [{stats_c['min']:.3f}, {stats_c['max']:.3f}]")
    print(f"  Mean ± Std: {stats_c['mean']:.3f} ± {stats_c['std']:.3f}")
    print(f"  Median: {stats_c['median']:.3f}")
    print(f"  Percentiles: P10={stats_c['p10']:.3f}, P90={stats_c['p90']:.3f}")
    if not exclude_zero:
        print(f"  Zeros: {stats_c['zero_count']:,} ({stats_c['zero_fraction']:.1%})")
    
    print(f"\n⏱️  Time Statistics{zero_suffix}{threshold_suffix}{transform_suffix}:")
    print(f"  Range: [{stats_t['min']:.3f}, {stats_t['max']:.3f}]")
    print(f"  Mean ± Std: {stats_t['mean']:.3f} ± {stats_t['std']:.3f}")
    print(f"  Median: {stats_t['median']:.3f}")
    print(f"  Percentiles: P10={stats_t['p10']:.3f}, P90={stats_t['p90']:.3f}")
    if not exclude_zero:
        print(f"  Zeros: {stats_t['zero_count']:,} ({stats_t['zero_fraction']:.1%})")
    if min_time_threshold is not None:
        print(f"  Min Threshold: {min_time_threshold:.1f}ns")
    if log_time_transform is not None:
        print(f"  Log Transform: {log_time_transform}")

def main():
    ap = argparse.ArgumentParser(description="Plot advanced histograms for input charge/time from HDF5")
    ap.add_argument("-p", "--path", required=True, help="Path to HDF5 file")
    ap.add_argument("--bins", type=int, default=200, help="Number of histogram bins")
    ap.add_argument("--chunk", type=int, default=1024, help="Batch chunk size for streaming")
    ap.add_argument("--range-charge", type=float, nargs=2, metavar=("MIN","MAX"),
                    help="Manual range for charge (e.g., 0 50)")
    ap.add_argument("--range-time", type=float, nargs=2, metavar=("MIN","MAX"),
                    help="Manual range for time (e.g., 0 20000)")
    ap.add_argument("--out", type=str, default="hist_input", help="Output PNG prefix")
    ap.add_argument("--logy", action="store_true", help="Use log-scale on y-axis")
    ap.add_argument("--logx", action="store_true", help="Use log-scale on x-axis")
    ap.add_argument("--no-stats", action="store_true", help="Hide statistical lines (mean, median, std)")
    ap.add_argument("--no-percentiles", action="store_true", help="Hide percentile lines (P10, P90, P25, P75)")
    ap.add_argument("--figsize", type=int, nargs=2, default=(12, 8), metavar=("WIDTH", "HEIGHT"),
                    help="Figure size (width, height)")
    ap.add_argument("--exclude-zero", action="store_true", help="Exclude zero values from histogram")
    ap.add_argument("--plot-both", action="store_true", help="Plot both all values and non-zero only")
    ap.add_argument("--min-time", type=float, help="Minimum time threshold (ns) - only plot time values above this")
    ap.add_argument("--log-time", choices=["log10", "ln"], default="log10",
                   help="Log transformation for time values: log10 (default) or ln")
    ap.add_argument("--pclip", type=float, nargs=2, default=(0.5, 99.5),
                    metavar=("LOW","HIGH"),
                    help="Percentiles for auto-range when range not given")
    args = ap.parse_args()

    # 기본 플롯 (모든 값 포함)
    if not args.exclude_zero or args.plot_both:
        plot_hist_pair(
            h5_path=args.path,
            bins=args.bins,
            chunk=args.chunk,
            range_charge=tuple(args.range_charge) if args.range_charge else None,
            range_time=tuple(args.range_time) if args.range_time else None,
            out_prefix=args.out,
            logy=args.logy,
            logx=args.logx,
            show_stats=not args.no_stats,
            show_percentiles=not args.no_percentiles,
            figsize=tuple(args.figsize),
            exclude_zero=False,
            plot_both=args.plot_both,
            min_time_threshold=args.min_time,
            log_time_transform=args.log_time,
            pclip=tuple(args.pclip),
        )
    
    # 0 제외 플롯
    if args.exclude_zero or args.plot_both:
        plot_hist_pair(
            h5_path=args.path,
            bins=args.bins,
            chunk=args.chunk,
            range_charge=tuple(args.range_charge) if args.range_charge else None,
            range_time=tuple(args.range_time) if args.range_time else None,
            out_prefix=args.out,
            logy=args.logy,
            logx=args.logx,
            show_stats=not args.no_stats,
            show_percentiles=not args.no_percentiles,
            figsize=tuple(args.figsize),
            exclude_zero=True,
            plot_both=args.plot_both,
            min_time_threshold=args.min_time,
            log_time_transform=args.log_time,
            pclip=tuple(args.pclip),
        )
    
    # 출력 파일 목록
    time_suffix = ""
    if args.min_time is not None:
        time_suffix += f"_min{args.min_time:.0f}"
    if args.log_time:
        time_suffix += f"_{args.log_time}"
    
    if args.plot_both:
        print(f"\n✅ Saved: {args.out}_charge.png, {args.out}_time{time_suffix}.png, {args.out}_charge_nonzero.png, {args.out}_time_nonzero{time_suffix}.png")
    elif args.exclude_zero:
        print(f"\n✅ Saved: {args.out}_charge_nonzero.png, {args.out}_time_nonzero{time_suffix}.png")
    else:
        print(f"\n✅ Saved: {args.out}_charge.png, {args.out}_time{time_suffix}.png")

if __name__ == "__main__":
    main()