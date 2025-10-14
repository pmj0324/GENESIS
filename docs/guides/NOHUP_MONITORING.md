# GENESIS 백그라운드 실행 및 모니터링 가이드

서버에서 GENESIS 학습을 백그라운드로 실행하고 모니터링하는 방법을 설명합니다.

## 📋 목차

1. [빠른 시작](#빠른-시작)
2. [스크립트 설명](#스크립트-설명)
3. [모니터링 방법](#모니터링-방법)
4. [문제 해결](#문제-해결)

---

## 🚀 빠른 시작

### 1. Task 생성
```bash
cd ~/GENESIS/GENESIS/tasks
python3 create_task.py my_experiment
```

### 2. 백그라운드로 실행
```bash
cd my_experiment
./start.sh
```

### 3. 상태 확인
```bash
./monitor.sh
```

### 4. 학습 중단
```bash
./stop.sh
```

---

## 📜 스크립트 설명

### `start.sh` - 백그라운드 실행

**기능:**
- nohup을 사용하여 학습을 백그라운드로 시작
- PID 파일 생성 (`.pid`)
- 중복 실행 방지
- 시작 성공 여부 확인

**사용법:**
```bash
./start.sh
```

**출력 정보:**
- 프로세스 ID (PID)
- 시작 시간
- 로그 파일 위치
- 모니터링 명령어

**생성 파일:**
- `.pid` : 프로세스 ID 저장
- `nohup.out` : 콘솔 출력
- `logs/*_training.txt` : 학습 로그

---

### `monitor.sh` - 상태 모니터링

**기능:**
- 프로세스 실행 여부 확인
- CPU/메모리 사용률 표시
- GPU 사용 현황 (nvidia-smi)
- 최근 로그 출력
- 실행 시간 표시

**사용법:**
```bash
# 한 번 확인
./monitor.sh

# 5초마다 자동 갱신
watch -n 5 ./monitor.sh

# 1초마다 자동 갱신 (빠른 모니터링)
watch -n 1 ./monitor.sh
```

**표시 정보:**
```
✅ Status: RUNNING

📊 Process Information:
   PID: 12345
   CPU: 95.2%
   Memory: 8.4%
   Runtime: 01:23:45
   Command: python3 ../../scripts/train.py --config config.yaml

🖥️  GPU Usage:
   GPU 0: NVIDIA GeForce RTX 3090
   Utilization: 98%
   Memory: 18432 MB / 24576 MB

📝 Recent Log Output:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Epoch 10/100 Summary
Train Loss: 0.012345
Val Loss: 0.011234
...
```

---

### `stop.sh` - 학습 중단

**기능:**
- Graceful shutdown (SIGTERM)
- 10초 대기 후 강제 종료 (SIGKILL)
- PID 파일 정리

**사용법:**
```bash
./stop.sh
```

**동작 순서:**
1. SIGTERM 전송 (graceful shutdown)
2. 10초 대기 (모델 저장 시간 확보)
3. 여전히 실행 중이면 SIGKILL (강제 종료)
4. `.pid` 파일 삭제

---

## 📊 모니터링 방법

### 1. 실시간 로그 확인

**콘솔 출력 (nohup.out):**
```bash
tail -f nohup.out
```

**학습 로그 (포맷된 출력):**
```bash
tail -f logs/*_training.txt
```

**특정 정보만 필터링:**
```bash
# Epoch 정보만 보기
tail -f logs/*_training.txt | grep "Epoch"

# Loss 정보만 보기
tail -f logs/*_training.txt | grep "Loss"

# 에러 정보만 보기
tail -f nohup.out | grep -i "error"
```

### 2. 프로세스 정보 확인

**PID 확인:**
```bash
cat .pid
```

**프로세스 상세 정보:**
```bash
ps -p $(cat .pid) -f
```

**프로세스 트리:**
```bash
pstree -p $(cat .pid)
```

**리소스 사용률:**
```bash
top -p $(cat .pid)
```

### 3. GPU 모니터링

**실시간 GPU 사용률:**
```bash
watch -n 1 nvidia-smi
```

**특정 프로세스 GPU 사용:**
```bash
nvidia-smi | grep $(cat .pid)
```

**GPU 메모리만 확인:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 4. 로그 분석

**최근 100줄 확인:**
```bash
tail -n 100 logs/*_training.txt
```

**에러 검색:**
```bash
grep -i "error" logs/*_training.txt
```

**특정 Epoch 검색:**
```bash
grep "Epoch 50" logs/*_training.txt
```

**Loss 추이 확인:**
```bash
grep "Val Loss:" logs/*_training.txt
```

---

## 🔍 유용한 명령어 모음

### 프로세스 관리

```bash
# PID로 프로세스 확인
ps -p $(cat .pid) -o pid,ppid,%cpu,%mem,etime,cmd

# 프로세스가 살아있는지 확인
ps -p $(cat .pid) > /dev/null 2>&1 && echo "Running" || echo "Not running"

# 프로세스 시작 시간 확인
ps -p $(cat .pid) -o lstart=

# 프로세스 실행 시간
ps -p $(cat .pid) -o etime=
```

### 로그 분석

```bash
# 로그 파일 크기
du -h logs/*_training.txt nohup.out

# 로그 라인 수
wc -l logs/*_training.txt nohup.out

# 특정 단어 개수
grep -c "Epoch" logs/*_training.txt

# 시간대별 로그 (마지막 1시간)
find logs/ -name "*_training.txt" -mmin -60 -exec tail -f {} \;
```

### GPU 모니터링

```bash
# GPU 온도 확인
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader

# GPU 전력 소비
nvidia-smi --query-gpu=power.draw --format=csv,noheader

# 모든 GPU 프로세스
nvidia-smi pmon

# GPU 메모리 사용률 그래프
watch -n 1 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
```

### 시스템 리소스

```bash
# 디스크 사용량
df -h .

# 디렉토리 크기
du -sh outputs/ logs/

# 메모리 사용량
free -h

# CPU 정보
lscpu
```

---

## 🛠️ 문제 해결

### 1. 프로세스가 시작되지 않음

**증상:**
```bash
./start.sh
❌ Failed to start training!
```

**확인:**
```bash
# 에러 메시지 확인
cat nohup.out

# Python 환경 확인
which python3
python3 --version

# 환경 변수 확인
echo $PYTHONPATH

# 설정 파일 확인
cat config.yaml
```

**해결:**
1. micromamba 환경 활성화 확인
2. 데이터 경로 확인
3. GPU 드라이버 확인

### 2. PID 파일이 남아있지만 프로세스 없음

**증상:**
```bash
./monitor.sh
❌ Status: NOT RUNNING (stale PID file)
```

**해결:**
```bash
# PID 파일 삭제
rm .pid

# 다시 시작
./start.sh
```

### 3. 학습이 멈춘 것 같음

**확인:**
```bash
# 프로세스 확인
./monitor.sh

# 최근 로그 확인
tail -n 50 logs/*_training.txt

# 프로세스 상태 확인
ps -p $(cat .pid) -o state=
```

**상태 코드:**
- `R` : Running (정상)
- `S` : Sleeping (I/O 대기, 정상)
- `D` : Uninterruptible sleep (디스크 I/O)
- `Z` : Zombie (비정상)
- `T` : Stopped (중단됨)

### 4. GPU 메모리 부족

**증상:**
```bash
CUDA out of memory
```

**확인:**
```bash
# GPU 메모리 사용 현황
nvidia-smi

# 다른 프로세스 확인
nvidia-smi pmon
```

**해결:**
1. 배치 크기 줄이기 (config.yaml)
2. 다른 GPU 프로세스 종료
3. GPU 재시작

### 5. 디스크 용량 부족

**확인:**
```bash
# 디스크 사용량
df -h .

# 큰 파일 찾기
du -ah outputs/ logs/ | sort -rh | head -20
```

**해결:**
```bash
# 오래된 체크포인트 삭제
find outputs/checkpoints/ -name "*.pth" -mtime +7 -delete

# 오래된 로그 압축
gzip logs/*.txt

# 불필요한 파일 정리
rm -f nohup.out.*
```

---

## 💡 팁 & 트릭

### 1. 여러 Task 동시 실행

```bash
# Task 1
cd tasks/experiment1
./start.sh

# Task 2
cd ../experiment2
./start.sh

# 모든 Task 모니터링
for task in tasks/*/; do
    echo "=== $task ==="
    cd "$task"
    ./monitor.sh
    cd -
done
```

### 2. 자동 재시작 (실패 시)

```bash
# auto_restart.sh
#!/bin/bash
while true; do
    ./start.sh
    wait $(cat .pid)
    echo "Training stopped. Restarting in 10 seconds..."
    sleep 10
done
```

### 3. 슬랙/이메일 알림

```bash
# notify.sh
#!/bin/bash
./monitor.sh > status.txt
if grep -q "Training completed" logs/*_training.txt; then
    # 슬랙 웹훅으로 알림
    curl -X POST -H 'Content-type: application/json' \
        --data "{'text':'Training completed!'}" \
        YOUR_SLACK_WEBHOOK_URL
fi
```

### 4. 원격 모니터링

```bash
# SSH로 원격 모니터링
ssh user@server "cd ~/GENESIS/GENESIS/tasks/my_experiment && ./monitor.sh"

# watch로 지속적 모니터링
watch -n 10 "ssh user@server 'cd ~/GENESIS/GENESIS/tasks/my_experiment && ./monitor.sh'"
```

---

## 📚 참고

- [GENESIS README](../../README.md)
- [Training Guide](./TRAINING.md)
- [Configuration Guide](./NORMALIZATION_CONFIG.md)
- [GPU Optimization](../../gpu_tools/README.md)

---

## 🆘 추가 도움말

문제가 해결되지 않으면:

1. **로그 확인**: `cat nohup.out logs/*_training.txt`
2. **이슈 등록**: GitHub Issues
3. **문의**: 프로젝트 관리자에게 연락

---

**마지막 업데이트**: 2024-01-14

