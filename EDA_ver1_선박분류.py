
# ================================================
# 대회장용 음파 EDA 코드 ver.1 (선박 분류)
# 수정할 곳: DATA_PATH, CLASSES 두 곳만!
# ================================================

!pip install librosa numpy matplotlib -q

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# ================================================
# ✅ 여기만 수정하면 됨
DATA_PATH = '/content/deepship/DeepShip-main'
CLASSES = ['Cargo', 'Passengership', 'Tanker', 'Tug']
MIN_DURATION = 10
# ================================================

print("=" * 50)
print("📊 음파 EDA 시작 (선박 분류용)")
print("=" * 50)

# 1. 이상한 폴더 확인
all_folders = [f for f in os.listdir(DATA_PATH)
               if os.path.isdir(os.path.join(DATA_PATH, f))]
unknown = [f for f in all_folders if f not in CLASSES]
if unknown:
    print(f"⚠️ 이상한 폴더 발견: {unknown} → 학습에서 제외")
else:
    print(f"✅ 이상한 폴더 없음")

# 2. 클래스 분포 확인
class_counts = {}
all_files = {}
for cls in CLASSES:
    cls_path = os.path.join(DATA_PATH, cls)
    if os.path.exists(cls_path):
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        class_counts[cls] = len(files)
        all_files[cls] = files

print(f"\n📊 클래스 분포:")
for cls, count in class_counts.items():
    print(f"  {cls}: {count}개")

max_count = max(class_counts.values())
min_count = min(class_counts.values())
min_cls = min(class_counts, key=class_counts.get)
ratio = max_count / min_count
if ratio > 3:
    print(f"⚠️ 클래스 불균형 감지 (비율 {ratio:.1f}:1)")
    print(f"⚠️ {min_cls} → Targeted Augmentation 필요")
else:
    print(f"✅ 클래스 균형 정상 (비율 {ratio:.1f}:1)")

total = sum(class_counts.values())
print(f"\n📊 총 데이터: {total}개")
if total < 500:
    print("⚠️ 데이터 매우 적음 → 전체 증강 필요")
elif total < 2000:
    print("⚠️ 데이터 적음 → 증강 권장")
else:
    print("✅ 데이터 충분")

# 3. 오디오 길이 확인
print(f"\n📊 오디오 길이 분포:")
durations = {}
short_files = []
for cls in CLASSES:
    cls_path = os.path.join(DATA_PATH, cls)
    cls_durations = []
    for file in all_files[cls]:
        filepath = os.path.join(cls_path, file)
        duration = librosa.get_duration(path=filepath)
        cls_durations.append(duration)
        if duration < MIN_DURATION:
            short_files.append(f"{cls}/{file} ({duration:.1f}초)")
    durations[cls] = cls_durations
    print(f"  {cls}: 평균 {np.mean(cls_durations):.1f}초 / 최소 {np.min(cls_durations):.1f}초 / 최대 {np.max(cls_durations):.1f}초")

if short_files:
    print(f"⚠️ {MIN_DURATION}초 미만 파일 {len(short_files)}개 → 제외 또는 패딩 필요")
else:
    print(f"✅ {MIN_DURATION}초 미만 파일 없음")

# 4. 샘플링 레이트 확인
sample_rates = set()
for cls in CLASSES:
    cls_path = os.path.join(DATA_PATH, cls)
    for file in all_files[cls][:3]:
        filepath = os.path.join(cls_path, file)
        _, sr = librosa.load(filepath, sr=None)
        sample_rates.add(sr)

print(f"\n📊 샘플링 레이트 종류: {sample_rates}")
if len(sample_rates) == 1:
    print(f"✅ 샘플링 레이트 통일: {list(sample_rates)[0]}Hz")
else:
    print(f"⚠️ 샘플링 레이트 불일치 → 22050Hz로 리샘플링 필요")

# 5. 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(class_counts.keys(), class_counts.values(),
            color="skyblue", edgecolor="black")
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Count")
for cls in CLASSES:
    axes[1].hist(durations[cls], bins=10, alpha=0.6, label=cls)
axes[1].set_title("Audio Duration Distribution")
axes[1].set_xlabel("Duration (seconds)")
axes[1].set_ylabel("Count")
axes[1].legend()
plt.tight_layout()
plt.show()

# 6. 전처리 전략 출력
print("\n" + "=" * 50)
print("📋 전처리 전략 요약")
print("=" * 50)
if unknown:
    print(f"1. {unknown} 폴더 학습에서 제외")
if ratio > 3:
    print(f"2. {min_cls} Targeted Augmentation")
if len(sample_rates) > 1:
    print(f"3. 전체 22050Hz로 리샘플링")
if short_files:
    print(f"4. {MIN_DURATION}초 미만 파일 제외")
if total < 2000:
    print(f"5. 전체 데이터 증강 권장")
print("=" * 50)
print("✅ EDA 완료 → 위 전략대로 전처리 시작!")
