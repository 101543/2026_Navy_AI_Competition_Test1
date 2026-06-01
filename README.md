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
    ├── train.py
    ├── predict.py
    └── utils.py
```

## 2. 데이터셋

이 프로젝트는 `torchvision.datasets.MNIST`를 사용합니다.
데이터는 처음 학습할 때 자동으로 `data/raw/`에 다운로드됩니다.

## 3. Colab에서 학습하기

1. 이 프로젝트를 GitHub에 업로드합니다.
2. Google Colab에서 `notebooks/colab_train.ipynb`를 엽니다.
3. 노트북의 안내대로 GitHub 저장소를 clone합니다.
4. 아래 명령으로 학습을 실행합니다.

```bash
python src/train.py --config configs/default.yaml
```

학습이 끝나면 모델 파일은 `outputs/models/best_model.pt`에 저장됩니다.

## 4. 예측 파일 만들기

```bash
python src/predict.py --config configs/default.yaml --checkpoint outputs/models/best_model.pt
```

결과 CSV는 `outputs/submissions/submission.csv`에 저장됩니다.

## 5. GitHub에 올리기

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
