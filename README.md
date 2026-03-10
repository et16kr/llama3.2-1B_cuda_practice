# llama3.2-1B_practice

CUDA 연습용 Llama 3.2 1B inference 프로젝트입니다.

- CPU 기준 forward를 먼저 제공하고, 학생이 CUDA kernel로 바꿔 가며 학습하는 형태를 목표로 합니다.
- Llama 계열 구조를 반영했습니다: `RMSNorm`, `RoPE`, `GQA`, `SwiGLU`.
- 기본 실행 진입점은 `scripts/run_text_generation.py`입니다.
- `main`은 Python wrapper가 내부적으로 호출하는 tokenized generation 실행기입니다.
- 입력 텍스트 파일은 `1줄 = 1개 요청` 형식입니다.
- 필요하면 기존처럼 pretokenized input 파일을 넣어 `forward -> logits` 모드로도 실행할 수 있습니다.

## 권장 모델

- Hugging Face `meta-llama/Llama-3.2-1B-Instruct`

기본 사용 경로인 Python wrapper는 Instruct 모델을 전제로 하며, tokenizer의 chat template를 그대로 사용합니다.

## 준비 사항

- CUDA Toolkit과 `nvcc`
- C++17 지원 컴파일러
- Python 3
- Python 패키지: `transformers`, `tokenizers`, `sentencepiece`, `jinja2`
- 로컬 모델 디렉터리

모델 디렉터리에는 최소한 아래 파일들이 있어야 합니다.

- `config.json`
- `model.safetensors`
- tokenizer 관련 파일들 (`tokenizer.json`, `tokenizer_config.json` 등)

## 프로젝트 구성

- `scripts/run_text_generation.py`: 기본 실행 진입점. text 입력 tokenize, `main` 호출, 결과 decode
- `scripts/build_hf_verified_request_bank.py`: Hugging Face로 응답 품질을 확인해 요청 뱅크 생성
- `scripts/sample_requests.py`: 요청 뱅크에서 원하는 개수만 랜덤 샘플링
- `src/main.cpp`: Python wrapper가 호출하는 tokenized generation/forward-only 엔트리 포인트
- `src/app_options.cpp`: CLI 파싱
- `src/generation.cpp`: batch generation 루프
- `src/model.cu`: Llama 파라미터 로딩과 전체 forward
- `src/layer.cu`: CPU 기준 연산 구현과 GPU TODO 함수
- `src/config.cpp`: `config.json` 로딩
- `data/`: 예시 요청 텍스트 파일

## 입력 파일 형식

`main`이 읽는 token batch 파일 형식:

- `int32 B`
- `int32 T`
- `int32 lengths[B]`
- `int32 token_ids[B*T]`

레거시 단순 형식인 `B, T, token_ids`도 계속 읽을 수 있습니다. 이 경우 모든 길이는 `T`로 간주합니다.

## 빌드

```bash
make
```

## 실행

일반적으로는 `main`을 직접 실행하지 않고 Python wrapper를 사용합니다.

### 1. 줄 단위 batch text generation

```bash
cat > ./data/requests.txt <<'EOF'
CUDA kernel 최적화 순서를 설명해줘.
RMSNorm과 LayerNorm의 차이를 요약해줘.
EOF

MODEL_DIR=/path/to/Llama-3.2-1B-Instruct \
INPUT_TXT=./data/requests.txt \
OUTPUT_TXT=./data/responses.txt \
make run
```

또는:

```bash
python3 ./scripts/run_text_generation.py \
        --model-dir /path/to/Llama-3.2-1B-Instruct \
        --input ./data/requests.txt \
        --output ./data/responses.txt
```

출력 파일은 각 요청과 그에 대응하는 답변을 delimiter와 함께 저장합니다.

### 2. 한 번만 요청하기

```bash
python3 ./scripts/run_text_generation.py \
        --model-dir /path/to/Llama-3.2-1B-Instruct \
        --prompt "CUDA 커널 최적화 순서를 설명해줘"
```

### 3. `main` 직접 실행

Python wrapper가 내부적으로 token file을 만들어 `main`을 호출합니다. 아래는 디버깅이나 실험용 직접 실행 예시입니다.

```bash
./main -m /path/to/Llama-3.2-1B-Instruct \
       --token-input ./data/prompts.bin \
       --token-output ./data/generated_tokens.bin
```

### 4. forward-only / logits 저장

```bash
./main -m /path/to/Llama-3.2-1B-Instruct \
       --token-input /path/to/your_tokens.bin \
       --logits-output ./data/logits.bin \
       -v
```

저장소에는 샘플 `bin` 파일을 포함하지 않습니다. 필요하면 Python wrapper가 만드는 token batch를 재사용하거나, 같은 포맷으로 직접 만들면 됩니다.

### 5. Hugging Face 기준 요청 뱅크 생성

Hugging Face 모델로 응답을 확인해, 품질이 불안정한 요청을 제외한 뱅크를 만들 수 있습니다.

```bash
python3 ./scripts/build_hf_verified_request_bank.py \
        --model-dir /path/to/Llama-3.2-1B-Instruct \
        --count 1200 \
        --output ./data/hf_verified_request_bank.txt
```

생성된 요청 뱅크에서 원하는 개수만 무작위로 추출하려면:

```bash
python3 ./scripts/sample_requests.py \
        --input ./data/hf_verified_request_bank.txt \
        --output ./data/requests.txt \
        --count 1024 \
        --seed 20260311
```

## 현재 상태

- `llama_forward()`는 CPU 기준 경로를 사용합니다.
- `*_gpu()` 함수는 CPU 기준 결과를 먼저 내고, 학생이 CUDA kernel로 교체할 수 있도록 TODO를 남겨 두었습니다.
- Python wrapper가 각 줄을 독립 요청으로 tokenize한 뒤, `main`은 tokenized batch에 대해 greedy decoding을 수행합니다.
- batch generation은 per-sequence valid length를 사용해 padding이 attention에 섞이지 않도록 처리합니다.
- weight loader는 `F32`, `F16`, `BF16` safetensors를 읽어 float로 변환합니다.

## 남은 연습 포인트

- `EmbeddingLookup_gpu`
- `RMSNorm_gpu`
- `Linear_gpu`
- `SplitHeads_gpu`
- `ApplyRoPE_gpu`
- `AttentionScoresGrouped_gpu`
- `ScaleMaskSoftmax_gpu`
- `AttentionContextGrouped_gpu`
- `MergeHeads_gpu`
- `ResidualAdd_gpu`
- `SiLU_gpu`
- `ElementwiseMul_gpu`
- `LMHead_gpu`

## 주의

- 현재 구현은 학습용 CPU reference 중심입니다. 긴 컨텍스트에서는 느립니다.
- 현재 로더와 CPU reference는 Llama 3 계열 `rope_scaling` 설정을 반영합니다.
- 출력 품질은 Python wrapper가 사용하는 tokenizer의 chat template와 local model 파일 상태에 따라 달라집니다.
