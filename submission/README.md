# 전체 슬라이드 이미지 (WSI) 분류 추론

이 저장소는 전체 슬라이드 이미지(WSI)에 대한 이진 분류 레이블을 예측하기 위한 Python 스크립트를 포함한다. 스크립트는 CSV 파일의 여러 WSI를 처리하고 예측 결과와 함께 확률을 출력한다.

## 개요

추론 파이프라인은 다음 단계를 수행한다:

1. **타일 추출**: 전체 슬라이드 이미지에서 조직 타일을 추출한다.
2. **타일 인코딩**: 사전 훈련된 타일 인코더 모델을 사용하여 타일을 인코딩한다.
3. **슬라이드 수준 예측**: 타일 특징을 집계하여 슬라이드 수준 예측을 수행한다.
4. **출력**: 예측 및 확률이 포함된 CSV 파일을 생성한다.

## 사전 요구사항

### 시스템 요구사항

- Python 3.8 이상
- CUDA 지원 GPU (더 빠른 추론을 위해 권장)
- 최소 16GB RAM (대형 슬라이드의 경우 32GB 권장)
- OpenSlide 라이브러리 (WSI 파일 읽기용)

### 필요한 파일

추론을 실행하기 전에 다음이 필요하다.

1. **모델 가중치**:
   - `tile_encoder.pth` - 사전 훈련된 타일 인코더 가중치
   - `slide_encoder.pt` - 사전 훈련된 슬라이드 인코더 가중치
2. 슬라이드 식별자가 포함된 **입력 CSV 파일** (아래 입력 형식 참조)
3. 디렉토리에 있는 **WSI 파일** (.svs 형식)

## 설치

```bash
# env.yml이 있는 경우 conda 환경 생성
conda env create -f env.yml
conda activate path_kids
pip install -r requirements.txt
```

## 입력 형식

### CSV 파일 구조

입력 CSV 파일에는 슬라이드 식별자가 포함된 최소 한 개의 열이 있어야 합니다. 기본 열 이름은 `SlideName`이지만 사용자 정의할 수 있습니다.

**예시 `query.csv`:**

```csv
SlideName
11_01_0606_PAS
12_02_0707_PAS
13_03_0808_PAS
```

**참고**: 스크립트는 WSI 파일을 찾을 때 각 슬라이드 이름에 자동으로 `.svs`를 추가한다.  `.svs`를 추가하면 에러가 발생할 수 있다. 

### WSI 파일 구성

다음과 같은 디렉토리 구조로 WSI 파일을 구성하세요:

```text
wsi_dir/
├── 11_01_0606_PAS.svs
├── 12_02_0707_PAS.svs
└── 13_03_0808_PAS.svs
```

## 사용법

### 기본 사용법

```bash
python inference.py \
    --csv_path query.csv \
    --wsi_dir ./wsi_dir \
    --tile_encoder_weights_path ./tile_encoder.pth \
    --slide_encoder_weights_path ./slide_encoder.pt \
    --output_dir ./output \
    --output_csv predictions.csv
```

### 사용 가능한 모든 인수

| 인수 | 유형 | 기본값 | 설명 |
|------|------|--------|------|
| `--csv_path` | str | `./query.csv` | 슬라이드 식별자가 포함된 입력 CSV 파일 경로 |
| `--wsi_dir` | str | `./wsi_dir` | WSI 파일(.svs 형식)이 포함된 디렉토리 |
| `--tile_encoder_weights_path` | str | `./tile_encoder.pth` | 타일 인코더 모델 가중치 경로 |
| `--slide_encoder_weights_path` | str | `./slide_encoder.pt` | 슬라이드 인코더 모델 가중치 경로 |
| `--slide_col_name` | str | `SlideName` | CSV에서 슬라이드 식별자를 포함하는 열 이름 |
| `--output_dir` | str | `output` | 출력 파일 디렉토리 |
| `--output_csv` | str | `analysis.csv` | 출력 CSV 파일 이름 |
| `--threshold` | float | `0.389` | 이진 분류를 위한 확률 임계값 (0-1) |
| `--amp` | str | `auto` | 자동 혼합 정밀도: `auto`, `true`, 또는 `false` |

### 예제 명령어

**예제 1: 최소 명령어 (기본값 사용)**

```bash
python inference.py
```

**예제 2: 사용자 정의 경로 및 임계값**

```bash
python inference.py \
    --csv_path /path/to/my_slides.csv \
    --wsi_dir /path/to/wsi_directory \
    --tile_encoder_weights_path /path/to/tile_encoder.pth \
    --slide_encoder_weights_path /path/to/slide_encoder.pt \
    --output_dir /path/to/results \
    --output_csv my_predictions.csv \
    --threshold 0.5
```

**예제 3: 사용자 정의 슬라이드 열 이름**

```bash
python inference.py \
    --csv_path slides.csv \
    --wsi_dir ./wsi \
    --slide_col_name SlideID \
    --tile_encoder_weights_path ./models/tile_encoder.pth \
    --slide_encoder_weights_path ./models/slide_encoder.pt
```

### 주의사항

- 다음 경고가 표시될 수 있습니다: 이러한 경고는 외부 패키지에서 발생하며 무시해도 됩니다.

```text
/opt/conda/envs/path_kids/lib/python3.12/site-packages/dask/dataframe/__init__.py:31: FutureWarning: The legacy Dask DataFrame implementation is deprecated and will be removed in a future version. Set the configuration option `dataframe.query-planning` to `True` or None to enable the new Dask Dataframe implementation and silence this warning.
  warnings.warn(
```

```text
/opt/conda/envs/path_kids/lib/python3.12/site-packages/xarray_schema/__init__.py:1: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  from pkg_resources import DistributionNotFound, get_distribution
```

```text
/opt/conda/envs/path_kids/lib/python3.12/site-packages/timm/models/layers/__init__.py:49: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
  warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
```

## 출력 형식

### CSV 출력

스크립트는 다음 열이 포함된 CSV 파일을 생성합니다:

| 열 | 설명 |
|-----|------|
| `ID` | 슬라이드 식별자 (.svs 확장자 포함) |
| `Predicted_Label` | 이진 예측 (0 또는 1) |
| `Predicted_Prob` | 확률 점수 (0.0 ~ 1.0) |

**예시 출력 (`predictions.csv`):**

```csv
ID,Predicted_Label,Predicted_Prob
11_01_0606_PAS.svs,1,0.8523
12_02_0707_PAS.svs,0,0.2341
13_03_0808_PAS.svs,1,0.6789
```

### 메트릭 JSON

추가로 출력 디렉토리에 요약 통계가 포함된 `metrics.json` 파일이 생성됩니다:

```json
{
  "csv_path": "/path/to/query.csv",
  "output_csv": "/path/to/output/predictions.csv",
  "num_rows": 100,
  "processed": 98,
  "skipped": 1,
  "failed": 1
}
```

## 출력 이해하기

### Predicted_Label

- **0**: 음성 클래스 예측
- **1**: 양성 클래스 예측

레이블은 `Predicted_Prob`를 임계값과 비교하여 결정됩니다:

- `Predicted_Prob >= threshold`인 경우 → `Predicted_Label = 1`
- `Predicted_Prob < threshold`인 경우 → `Predicted_Label = 0`

### Predicted_Prob

- 더 높은 값은 양성 클래스에 대한 더 높은 신뢰도를 나타냅니다
- 0.5에 가까운 값은 불확실한 예측을 나타냅니다

## 문제 해결

### 일반적인 문제

**1. FileNotFoundError: Path not found**

- **해결 방법**: 모든 파일 경로가 올바른지 확인하고 파일이 존재하는지 확인하세요
- CSV 경로, WSI 디렉토리 및 모델 가중치 경로를 확인하세요

**2. Slide not found: [slide_name].svs**

- **해결 방법**: WSI 파일이 지정된 `wsi_dir` 디렉토리에 있는지 확인하세요
- CSV의 슬라이드 이름이 WSI 파일 이름과 일치하는지 확인하세요 (.svs 확장자 제외)
- 파일 권한을 확인하세요

**3. CUDA out of memory**

- **해결 방법**: 배치 크기를 줄이거나 CPU 모드를 사용하세요
- `--amp false`를 설정하여 혼합 정밀도를 비활성화하세요
- 한 번에 더 적은 슬라이드를 처리하세요

**4. Import errors**

- **해결 방법**: 모든 종속성이 설치되어 있는지 확인: `pip install -r requirements.txt`
- Python 버전이 3.8 이상인지 확인하세요

**5. OpenSlide errors**

- **해결 방법**: OpenSlide 라이브러리를 설치하세요:

  ```bash
  # Ubuntu/Debian
  sudo apt-get install openslide-tools
  
  # macOS
  brew install openslide
  
  # conda
  conda install -c conda-forge openslide
  
  # 그 다음 Python 바인딩 설치
  pip install openslide-python
  conda install -c conda-forge openslide-python
  ```

### 성능 팁

1. **GPU 사용**: 스크립트는 사용 가능한 경우 자동으로 GPU를 사용합니다. CPU 모드를 강제하려면 `--amp false`를 설정하세요
2. **배치 처리**: 스크립트는 슬라이드를 순차적으로 처리합니다. 대용량 데이터셋의 경우 CSV를 더 작은 배치로 분할하는 것을 고려하세요
3. **메모리**: 각 슬라이드는 독립적으로 처리되므로 메모리 사용량은 데이터셋 크기가 아닌 슬라이드 크기에 따라 확장됩니다

## 기술 세부사항

### 모델 아키텍처

- **타일 인코더**: 개별 이미지 타일에서 특징을 추출합니다
- **슬라이드 인코더**: 타일 특징을 집계하여 슬라이드 수준 예측을 수행합니다
- **분류기**: 최종 분류 레이어

### 처리 파이프라인

1. 조직 감지 및 타일링 (lazyslide 사용)
2. 타일 특징 추출 (타일 인코더)
3. 슬라이드 수준 집계 (슬라이드 인코더)
4. 분류 (분류기)

### 지원 형식

- **입력 WSI**: .svs (Aperio ScanScope), .ndpi (Hamamatsu), .vms/.vmu (Leica), .scn (Aperio), .mrxs (MIRAX)
- **입력 CSV**: 슬라이드 식별자가 포함된 표준 CSV 형식
- **출력 CSV**: 예측이 포함된 표준 CSV 형식

## 인용

이 코드를 사용하는 경우 관련 출판물을 인용해 주세요 (해당되는 경우).

## 지원

문제나 질문이 있는 경우:

1. 위의 문제 해결 섹션을 확인하세요
2. 콘솔 출력의 오류 메시지를 검토하세요
3. 모든 입력 파일과 경로가 올바른지 확인하세요

## 라이선스

[해당되는 경우 라이선스 지정]
