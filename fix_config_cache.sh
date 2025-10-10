#!/bin/bash
# Config 캐시 문제 해결 스크립트

echo "================================================================================"
echo "🔧 Config 캐시 문제 해결"
echo "================================================================================"

echo ""
echo "Step 1: Python 캐시 완전 제거"
echo "--------------------------------------------------------------------------------"
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "✅ Python cache cleared"

echo ""
echo "Step 2: config.py 확인"
echo "--------------------------------------------------------------------------------"
echo "현재 config.py의 label_scales:"
grep "label_scales.*Tuple" config.py | head -1
echo ""

echo "Step 3: default.yaml 확인"
echo "--------------------------------------------------------------------------------"
echo "현재 default.yaml의 label_scales:"
grep "label_scales:" configs/default.yaml
echo ""

echo "Step 4: Python에서 직접 확인"
echo "--------------------------------------------------------------------------------"
python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

# config 모듈이 캐시되어 있다면 제거
if 'config' in sys.modules:
    del sys.modules['config']

# 새로 import
from config import ModelConfig

config = ModelConfig()
print(f"Python에서 읽은 label_scales: {config.label_scales}")
print(f"예상 값: (5e7, 1.0, 1.0, 600.0, 550.0, 550.0)")

if config.label_scales == (5e7, 1.0, 1.0, 600.0, 550.0, 550.0):
    print("✅ 올바른 값!")
else:
    print("❌ 여전히 이전 값!")
    print(f"   실제: {config.label_scales}")
PYEOF

echo ""
echo "================================================================================"
echo "다음 단계:"
echo "================================================================================"
echo ""
echo "1. 위 출력에서 label_scales 값 확인"
echo "2. 만약 여전히 이전 값이면:"
echo "   → config.py 파일을 직접 열어서 수동 확인"
echo "   → 파일 저장이 제대로 안되었을 수 있음"
echo ""
echo "3. 올바른 값이면:"
echo "   → 학습 재시작"
echo ""
echo "================================================================================"
