# Data

MNIST 원본 데이터는 `torchvision`이 `data/raw/`에 자동 다운로드합니다.

아래 명령을 실행하면 실제 이미지 파일과 CSV가 `data/processed/`에 생성됩니다.

```bash
python src/prepare_mnist.py --config configs/default.yaml
```

생성되는 구조:

```text
data/processed/
├── train.csv
├── val.csv
├── test.csv
├── train_images/
├── val_images/
└── test_images/
```

대용량 데이터는 GitHub에 올리지 않는 것이 좋기 때문에 `data/raw/`와 `data/processed/`는 `.gitignore`에 포함되어 있습니다.
