# MNIST Competition Template

MNIST 이미지 분류를 AI 경진대회 프로젝트 구조로 연습하기 위한 템플릿입니다.
학습은 Google Colab에서 실행하는 흐름을 기준으로 구성했습니다.

## 1. 프로젝트 구조

```text
mnist_competition_template/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── default.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── notebooks/
│   └── colab_train.ipynb
├── outputs/
│   ├── models/
│   └── submissions/
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── prepare_mnist.py
    ├── train.py
    ├── predict.py
    └── utils.py
```

## 2. 데이터셋

이 프로젝트는 `torchvision.datasets.MNIST`를 내려받은 뒤 실제 이미지 파일과 CSV로 변환해서 사용합니다.
경진대회에서 자주 쓰는 `이미지 폴더 + 라벨 CSV` 구조를 연습하기 위한 방식입니다.

```bash
python src/prepare_mnist.py --config configs/default.yaml
```

실행 후 데이터 구조는 아래처럼 만들어집니다.

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
├── train_images/
├── val_images/
└── test_images/
```

CSV 예시는 다음과 같습니다.

```text
id,image_path,label
1,train_images/00001.png,5
2,train_images/00002.png,0
```

`test.csv`에는 정답 라벨이 없고, 예측 후 `outputs/submissions/submission.csv`를 만듭니다.
이미지 데이터는 크기가 커질 수 있으므로 GitHub에는 올리지 않고 Colab이나 로컬 환경에서 생성합니다.

## 3. Colab에서 학습하기

Colab에 이 코드 폴더를 직접 업로드하거나 GitHub 저장소를 clone한 뒤 아래 순서대로 실행합니다.

```bash
pip install -r requirements.txt
python src/prepare_mnist.py --config configs/default.yaml
python src/train.py --config configs/default.yaml
python src/predict.py --config configs/default.yaml --checkpoint outputs/models/best_model.pt
```

학습이 끝나면 모델 파일은 `outputs/models/best_model.pt`에 저장됩니다.
예측 결과 CSV는 `outputs/submissions/submission.csv`에 저장됩니다.

## 4. GitHub에 올리기

터미널에서 프로젝트 폴더로 이동한 뒤 아래 명령을 실행하세요.

```bash
git init
git add .
git commit -m "Initial MNIST competition template"
git branch -M main
git remote add origin https://github.com/YOUR_ID/YOUR_REPOSITORY.git
git push -u origin main
```

`YOUR_ID`와 `YOUR_REPOSITORY`는 본인의 GitHub 계정과 저장소 이름으로 바꾸면 됩니다.
