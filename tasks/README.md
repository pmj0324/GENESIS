# GENESIS Tasks 디렉토리

이 디렉토리는 GENESIS 실험 task들을 관리하는 곳입니다. 각 task는 독립적인 실험 환경으로 설정, 로그, 체크포인트를 포함합니다.

---

## 📋 목차

- [빠른 시작](#빠른-시작)
- [Task 생성](#task-생성)
- [실행 방법](#실행-방법)
- [모니터링](#모니터링)
- [Task 구조](#task-구조)
- [유용한 명령어](#유용한-명령어)

---

## 🚀 빠른 시작

### 1. 새로운 Task 생성

```bash
# 기본 설정으로 생성
python3 create_task.py my_experiment

# 특정 설정으로 생성
python3 create_task.py my_experiment --config testing
python3 create_task.py my_experiment --config cosine
```

### 2. 실행

**Foreground (로컬 테스트):**
```bash
cd my_experiment
./run.sh
```

**Background (서버 학습):**
```bash
cd my_experiment
./start.sh           # 백그라운드 실행
./monitor.sh         # 상태 확인
./stop.sh            # 중단
```

---

## 📁 Task 생성

### 기본 사용법

```bash
python3 create_task.py <task_name> [--config CONFIG]
```

### 예시

```bash
# 기본 설정 (default.yaml)
python3 create_task.py experiment_01

# 빠른 테스트용 (testing.yaml - 10% 데이터)
python3 create_task.py quick_test --config testing

# Cosine annealing 스케줄러
python3 create_task.py cosine_exp --config cosine

# C-DiT 모델
python3 create_task.py c_dit_test --config models/c-dit
```

### 사용 가능한 Config 템플릿

```bash
# 기본 설정
- default.yaml          # 표준 설정
- testing.yaml          # 빠른 테스트 (10% 데이터)
- cosine.yaml           # Cosine annealing 스케줄러

# 모델별 설정
- models/c-dit.yaml     # Classifier-Attention DiT
- models/small_model.yaml  # 작은 모델 (빠른 실험)

# 학습 설정
- training/plateau.yaml      # ReduceLROnPlateau
- training/cosine_annealing.yaml  # Cosine annealing
- training/step.yaml         # Step LR
- training/linear.yaml       # Linear LR

# 데이터 변환
- data/ln_transform.yaml     # ln(1+x) 변환
- data/log10_transform.yaml  # log10(1+x) 변환
```

---

## 🖥️ 실행 방법

### Foreground 실행 (로컬)

화면에서 직접 확인하며 실행:

```bash
cd my_experiment
./run.sh
```

**특징:**
- 실시간 출력 확인
- Ctrl+C로 중단
- 터미널 종료 시 학습도 중단

**사용 시기:**
- 로컬 머신에서 테스트
- 설정 확인
- 디버깅

---

### Background 실행 (서버)

nohup으로 백그라운드 실행:

```bash
cd my_experiment
./start.sh
```

**특징:**
- SSH 연결 끊어도 계속 실행
- PID 파일로 프로세스 추적
- 중복 실행 자동 방지

**출력:**
```
🚀 Starting GENESIS training in background...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ Training started successfully!

📊 Job Information:
   PID: 12345
   Task: my_experiment
   Time: 2024-01-14 02:54:30

📝 Log files:
   Console output: nohup.out
   Training log: logs/*_training.txt

💡 Monitoring commands:
   ./monitor.sh              # Show status
   ./stop.sh                 # Stop training
   tail -f nohup.out         # Watch console output
   tail -f logs/*_training.txt  # Watch training log
```

**사용 시기:**
- 서버에서 장시간 학습
- 여러 실험 동시 실행
- 안정적인 학습 환경

---

## 📊 모니터링

### `monitor.sh` - 통합 모니터링

전체 상태를 한눈에 확인:

```bash
./monitor.sh
```

**표시 정보:**
- ✅ 프로세스 상태 (RUNNING/NOT RUNNING)
- 📊 CPU/메모리 사용률
- 🖥️ GPU 사용 현황 (nvidia-smi)
- ⏱️ 실행 시간
- 📝 최근 로그 (Epoch/Loss)

**출력 예시:**
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 GENESIS Training Status
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Status: RUNNING

📊 Process Information:
   PID: 12345
   CPU: 95.2%
   Memory: 8.4%
   Runtime: 01:23:45
   Command: python3 ../../scripts/train.py --config config.yaml

🖥️  GPU Usage:
   GPU 0: NVIDIA A100
   Utilization: 98%
   Memory: 18432 MB / 40960 MB

📝 Recent Log Output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 10/100 Summary
Train Loss: 0.012345
Val Loss: 0.011234
Learning Rate: 0.0001
```

### 자동 갱신 모니터링

5초마다 자동으로 상태 확인:

```bash
watch -n 5 ./monitor.sh
```

1초마다 빠른 갱신:

```bash
watch -n 1 ./monitor.sh
```

### 로그 실시간 확인

**콘솔 출력 (모든 출력):**
```bash
tail -f nohup.out
```

**학습 로그 (포맷된 출력):**
```bash
tail -f logs/*_training.txt
```

**특정 정보만 필터링:**
```bash
# Epoch 정보만
tail -f logs/*_training.txt | grep "Epoch"

# Loss 정보만
tail -f logs/*_training.txt | grep "Loss"

# 에러만
tail -f nohup.out | grep -i "error"
```

---

## 🛑 학습 중단

### Graceful Shutdown

정상적으로 학습을 중단하고 모델을 저장:

```bash
./stop.sh
```

**동작:**
1. SIGTERM 신호 전송 (정상 종료)
2. 10초 대기 (모델 저장 시간)
3. 필요시 SIGKILL (강제 종료)
4. `.pid` 파일 정리

**출력:**
```
🛑 Stopping GENESIS training...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Found running process: PID 12345

Sending SIGTERM (graceful shutdown)...
..........
✅ Process stopped gracefully
```

---

## 📂 Task 구조

각 task는 다음과 같은 구조를 가집니다:

```
my_experiment/
├── .pid                    # 프로세스 ID (실행 중에만 존재)
├── nohup.out              # 콘솔 출력 (stderr + stdout)
├── config.yaml            # 실험 설정
├── README.md              # Task 설명서
│
├── run.sh                 # 일반 실행 스크립트
├── start.sh               # 백그라운드 실행 스크립트
├── monitor.sh             # 모니터링 스크립트
├── stop.sh                # 중단 스크립트
│
├── logs/                  # 학습 로그
│   └── icecube_diffusion_*_training.txt
│
└── outputs/               # 학습 결과
    ├── checkpoints/       # 저장된 모델
    │   ├── best_model.pth
    │   └── checkpoint_epoch_*.pth
    ├── samples/           # 생성된 샘플
    ├── evaluation/        # 평가 결과
    │   └── final_evaluation/
    └── plots/             # 생성된 그래프
        ├── training_curves.png    # 학습 곡선 (PNG)
        └── training_curves.pdf    # 학습 곡선 (PDF)
```

---

## 🔧 유용한 명령어

### 프로세스 관리

```bash
# PID 확인
cat .pid

# 프로세스 상세 정보
ps -p $(cat .pid) -f

# 프로세스 트리
pstree -p $(cat .pid)

# 프로세스가 실행 중인지 확인
ps -p $(cat .pid) > /dev/null 2>&1 && echo "Running" || echo "Not running"

# 실행 시간 확인
ps -p $(cat .pid) -o etime=

# CPU/메모리 모니터링
top -p $(cat .pid)
```

### GPU 모니터링

```bash
# 실시간 GPU 사용률
watch -n 1 nvidia-smi

# 특정 프로세스 GPU 사용
nvidia-smi | grep $(cat .pid)

# GPU 메모리만
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# GPU 온도
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# 모든 GPU 프로세스
nvidia-smi pmon
```

### 로그 분석

```bash
# 로그 파일 크기
du -h logs/*_training.txt nohup.out

# 전체 라인 수
wc -l logs/*_training.txt

# 특정 단어 개수
grep -c "Epoch" logs/*_training.txt

# 최근 100줄
tail -n 100 logs/*_training.txt

# 에러 검색
grep -i "error" logs/*_training.txt nohup.out

# Val Loss 추이
grep "Val Loss:" logs/*_training.txt

# 특정 Epoch 검색
grep "Epoch 50" logs/*_training.txt
```

### 디스크 관리

```bash
# Task 전체 크기
du -sh .

# 각 디렉토리 크기
du -sh */ | sort -rh

# 큰 파일 찾기
find . -type f -size +100M -exec ls -lh {} \;

# 디스크 사용량
df -h .

# 오래된 체크포인트 삭제 (7일 이상)
find outputs/checkpoints/ -name "*.pth" -mtime +7 -delete

# 로그 압축
gzip logs/*.txt
```

### 여러 Task 관리

```bash
# 모든 Task 목록
ls -d */

# 실행 중인 Task 찾기
for task in */; do
    if [ -f "$task/.pid" ]; then
        echo "Running: $task (PID: $(cat $task/.pid))"
    fi
done

# 모든 Task 상태 확인
for task in */; do
    echo "=== $task ==="
    cd "$task"
    if [ -f "monitor.sh" ]; then
        ./monitor.sh
    fi
    cd ..
done

# 모든 실행 중인 Task 중단
for task in */; do
    if [ -f "$task/.pid" ]; then
        echo "Stopping $task..."
        cd "$task"
        ./stop.sh
        cd ..
    fi
done
```

---

## 💡 팁 & 트릭

### 1. Config 미리보기

Task 생성 전에 설정 확인:

```bash
cat ../configs/default.yaml
cat ../configs/testing.yaml
```

### 2. Config 수정

Task 생성 후 설정 변경:

```bash
cd my_experiment
vim config.yaml
```

주요 수정 항목:
- `data.batch_size`: 배치 크기
- `training.num_epochs`: 에포크 수
- `training.learning_rate`: 학습률
- `model.hidden`: 모델 크기

### 3. 원격 모니터링

SSH로 원격에서 상태 확인:

```bash
ssh user@server "cd ~/GENESIS/GENESIS/tasks/my_experiment && ./monitor.sh"
```

자동 갱신:

```bash
watch -n 10 "ssh user@server 'cd ~/GENESIS/GENESIS/tasks/my_experiment && ./monitor.sh'"
```

### 4. 실험 노트

각 task의 README.md에 실험 노트 작성:

```bash
cd my_experiment
vim README.md
```

기록할 내용:
- 실험 목적
- 변경한 하이퍼파라미터
- 예상 결과
- 실제 결과
- 문제점 및 해결 방법

### 5. 결과 비교

여러 실험 결과 비교:

```bash
# Val Loss 비교
grep "Val Loss:" */logs/*_training.txt | tail -n 1

# 최고 성능 찾기
for task in */; do
    loss=$(grep "New best model" $task/logs/*_training.txt 2>/dev/null | tail -n 1 | grep -oP '\d+\.\d+' | head -1)
    if [ ! -z "$loss" ]; then
        echo "$task: $loss"
    fi
done | sort -t: -k2 -n
```

### 6. 자동 재시작

학습 실패 시 자동 재시작 (주의해서 사용):

```bash
# auto_restart.sh
#!/bin/bash
cd my_experiment
while true; do
    ./start.sh
    wait $(cat .pid)
    echo "Training stopped. Restarting in 10 seconds..."
    sleep 10
done
```

---

## 🆘 문제 해결

### 1. Task 생성 실패

**증상:** "Task already exists" 에러

**해결:**
```bash
# 다른 이름 사용 또는 기존 task 삭제
rm -rf my_experiment
python3 create_task.py my_experiment
```

### 2. 시작 실패

**증상:** `./start.sh`가 실패

**확인:**
```bash
# 에러 메시지 확인
cat nohup.out

# Python 환경 확인
which python3
python3 --version

# 데이터 경로 확인
grep h5_path config.yaml
```

### 3. PID 파일 문제

**증상:** "stale PID file" 경고

**해결:**
```bash
# PID 파일 삭제
rm .pid

# 다시 시작
./start.sh
```

### 4. GPU 메모리 부족

**증상:** CUDA out of memory

**해결:**
```bash
# 1. 배치 크기 줄이기
vim config.yaml
# data.batch_size: 512 → 256

# 2. 다른 프로세스 확인 및 종료
nvidia-smi
kill <PID>

# 3. GPU 재시작 (주의!)
sudo nvidia-smi --gpu-reset
```

### 5. 디스크 용량 부족

**해결:**
```bash
# 오래된 체크포인트 삭제
find outputs/checkpoints/ -name "checkpoint_epoch_*.pth" -mtime +7 -delete

# 로그 압축
gzip logs/*.txt nohup.out

# 불필요한 task 삭제
rm -rf old_experiment
```

---

## 📚 추가 문서

- **완벽한 가이드**: [docs/guides/NOHUP_MONITORING.md](../docs/guides/NOHUP_MONITORING.md)
- **Training Guide**: [docs/guides/TRAINING.md](../docs/guides/TRAINING.md)
- **Configuration Guide**: [docs/guides/NORMALIZATION_CONFIG.md](../docs/guides/NORMALIZATION_CONFIG.md)
- **GPU Optimization**: [gpu_tools/README.md](../gpu_tools/README.md)

---

## 📝 예시 워크플로우

### 실험 시작부터 결과 확인까지

```bash
# 1. Task 생성
cd ~/GENESIS/GENESIS/tasks
python3 create_task.py experiment_20240114

# 2. 설정 확인 및 수정 (필요시)
cd experiment_20240114
vim config.yaml

# 3. 백그라운드 실행
./start.sh

# 4. 실시간 모니터링 (별도 터미널)
watch -n 5 ./monitor.sh

# 5. 로그 확인 (또 다른 터미널)
tail -f logs/*_training.txt

# 6. SSH 접속 종료 (학습은 계속됨)
exit

# --- 나중에 다시 접속 ---

# 7. 상태 확인
cd ~/GENESIS/GENESIS/tasks/experiment_20240114
./monitor.sh

# 8. 결과 확인
cat logs/*_training.txt | grep "New best model"
ls outputs/checkpoints/

# 9. 샘플링 (학습 완료 후)
python3 ../../scripts/sample.py --checkpoint outputs/checkpoints/best_model.pth

# 10. 실험 노트 작성
vim README.md
```

---

**마지막 업데이트**: 2024-01-14  
**버전**: 1.0
