# FastFace v7 Architecture Documentation

## Overview

FastFace v7은 **얼굴 정체성 보존 스타일 트랜스퍼 시스템**이다. 사용자가 제공한 얼굴 이미지의 identity를 보존하면서 스타일 이미지의 배경, 포즈, 색감을 반영한 새로운 이미지를 생성한다.

### 핵심 목표

```
[Face Image] + [Style Image] + [Text Prompt] --> [Generated Image]
     |               |               |
  얼굴 정체성     포즈/배경/색감    추가 수정 지시
```

### 버전 히스토리

| Version | Key Changes |
|---------|-------------|
| v0 | 논문 원본 구현 |
| v1 | Mac MPS 지원, Web UI |
| v2 | RealVisXL, 비동기 Task 시스템 |
| v3 | Batch-Wise DCG |
| v4 | CLIP Blending (Identity Loss 해결) |
| v5 | ControlNet Depth 통합 (구조 보존) |
| v6 | Face Masking (스타일 이미지 얼굴 충돌 해결) |
| v7 | 프론트엔드 파라미터 완전 제어, 마스크 미리보기 |

---

## System Architecture

### 전체 파이프라인 구조

```
+-------------------------------------------------------------------------+
|                         FastFace v7 Pipeline                             |
+-------------------------------------------------------------------------+
|                                                                          |
|  [Face Image]                          [Style Image]                     |
|       |                                      |                           |
|       v                                      v                           |
|  +------------+                      +---------------+                   |
|  | InsightFace|                      | InsightFace   |                   |
|  | Detection  |                      | Face Check    |                   |
|  +-----+------+                      +-------+-------+                   |
|        |                                     |                           |
|        v                                     v                           |
|  +------------+                      +---------------+                   |
|  | Face ID    |                      | Face Detected?|                   |
|  | Embedding  |                      +-------+-------+                   |
|  | (512-dim)  |                              |                           |
|  +-----+------+                    +---------+---------+                 |
|        |                           |                   |                 |
|        |                           v                   v                 |
|        |                    [YES: v6+ Flow]    [NO: Standard]            |
|        |                           |                   |                 |
|        |                           v                   |                 |
|        |                    +------------+             |                 |
|        |                    | SegFormer  |             |                 |
|        |                    | Face Parse |             |                 |
|        |                    +-----+------+             |                 |
|        |                          |                    |                 |
|        |                          v                    |                 |
|        |                    +------------+             |                 |
|        |                    | Face+Hair  |             |                 |
|        |                    | Mask       |             |                 |
|        |                    +-----+------+             |                 |
|        |                          |                    |                 |
|        |           +--------------+-------------+      |                 |
|        |           |              |             |      |                 |
|        |           v              v             v      |                 |
|        |    +-----------+  +-----------+ +-----------+ |                 |
|        |    | Blur Face |  | Mask Depth| | Save for  | |                 |
|        |    | Region    |  | Map Face  | | Preview   | |                 |
|        |    +-----------+  +-----------+ +-----------+ |                 |
|        |           |              |                    |                 |
|        |           v              v                    v                 |
|        |    [Masked Style] [Masked Depth]    [Original Style]           |
|        |           |              |                    |                 |
|        +-----------|--------------|--------------------+                 |
|                    |              |                                      |
|        +-----------+              |                                      |
|        |                         |                                      |
|        v                         v                                      |
|  +------------+           +------------+                                |
|  | CLIP       |           | ControlNet |                                |
|  | Blending   |           | Depth      |                                |
|  +-----+------+           +-----+------+                                |
|        |                        |                                       |
|        v                        v                                       |
|  +------------------------------------------------------------------+  |
|  |                    SDXL UNet + IP-Adapter + DCG                   |  |
|  |                                                                    |  |
|  |  [Text Embed] -----> Text Cross-Attention                          |  |
|  |                              |                                      |  |
|  |  [FaceID Embed] --> IP-Adapter FaceID ---> + <--- DCG Guidance    |  |
|  |                                            |                       |  |
|  |  [Blended CLIP] --> CLIP Shortcut --------+                       |  |
|  |                                            |                       |  |
|  |  [Init Latents] --> img2img (denoising) ---+                      |  |
|  |                                            |                       |  |
|  |  [Depth Map] ----> ControlNet Conditioning-+                      |  |
|  |                                                                    |  |
|  +------------------------------------------------------------------+  |
|                              |                                          |
|                              v                                          |
|                       +------------+                                    |
|                       | VAE Decode |                                    |
|                       +-----+------+                                    |
|                             |                                           |
|                             v                                           |
|                    [Generated Image]                                    |
|                                                                         |
+-------------------------------------------------------------------------+
```

---

## Core Models

### 1. Base Model: SDXL (Stable Diffusion XL)

**구조**: UNet2DConditionModel
**해상도**: 1024x1024 (기본)
**정밀도**: float16 (CUDA/MPS)

#### 지원 모델 Variants

| Model Name | Full Name | Steps | 특징 |
|------------|-----------|-------|------|
| `hyper` | Hyper-SD + RealVisXL | 1-8 | 권장, 빠른 생성 |
| `realvis` | RealVisXL V5.0 | 20-50 | 고품질 실사 |
| `lightning` | SDXL-Lightning | 2-8 | 빠른 생성 |
| `lcm` | LCM-LoRA SDXL | 4-8 | 일관성 모델 |
| `turbo` | SDXL-Turbo | 1-4 | 512x512, 초고속 |
| `base` | SDXL 1.0 | 20-50 | 표준 |

**로드 방식** (`src/sdxl_custom_pipeline.py`):
```python
# Hyper 모델 예시
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
# Hyper-SD LoRA 병합
pipe.load_lora_weights("ByteDance/Hyper-SD", weight_name="Hyper-SDXL-8steps-lora.safetensors")
pipe.fuse_lora(lora_scale=0.6)
```

### 2. IP-Adapter FaceID Plus v2

**용도**: 얼굴 정체성(identity) 주입
**구조**: ImageProjection + Cross-Attention Injection
**입력 차원**: `[batch, num_images, seq_len, hidden_dim]` = `[3-4, 1, 257, 1280]`

#### 작동 원리

```
[얼굴 이미지]
     |
     v
+------------------+
| InsightFace      |
| buffalo_l        |
+--------+---------+
         |
         v
[512-dim normed_embedding]  <-- 얼굴 고유 벡터
         |
         v
+------------------+
| ImageProjection  |  <-- IP-Adapter 학습된 투영층
+--------+---------+
         |
         v
[IP-Adapter Embedding]
         |
         v
UNet Cross-Attention에 주입
```

**핵심 파라미터**:
- `ip_adapter_scale` (ips): FaceID 강도 (기본 0.8)
- 높을수록 원본 얼굴에 가깝게 생성

### 3. ControlNet Depth (v5+)

**용도**: 스타일 이미지의 공간 구조 보존
**모델**: `diffusers/controlnet-depth-sdxl-1.0`
**입력**: MiDaS로 추출한 Depth Map

#### v6+ Face Masking과 연동

```
[Style Image]
     |
     v
+------------------+
| MiDaS Depth      |
| Estimator        |
+--------+---------+
         |
         v
[Raw Depth Map]
         |
         v (Face Detected?)
         |
    +----+----+
    |         |
    v         v
[Masked]   [Original]
    |         |
    +----+----+
         |
         v
[Final Depth for ControlNet]
```

**Depth Masking 이유**:
- 스타일 이미지 얼굴의 윤곽/형태가 ControlNet을 통해 강제됨
- 이를 방지하기 위해 얼굴 영역 depth를 blur 처리
- 결과: 배경/몸 구조는 유지, 얼굴 형태는 FaceID가 자유롭게 생성

**관련 파라미터**:
- `controlnet_scale`: 구조 유지 강도 (기본 0.4)
- `depth_blur_radius`: 얼굴 영역 blur 정도 (기본 80)

### 4. SegFormer Face Parser (v6+)

**용도**: 스타일 이미지에서 얼굴+머리카락 영역 분할
**모델**: `jonathandinu/face-parsing`
**아키텍처**: SegformerForSemanticSegmentation

#### 세그멘테이션 클래스

```python
FACE_PARSE_LABELS = {
    0: 'background',
    1: 'skin',
    2: 'nose',
    3: 'eyeglasses',
    4: 'left_eye',
    5: 'right_eye',
    6: 'left_eyebrow',
    7: 'right_eyebrow',
    8: 'left_ear',
    9: 'right_ear',
    10: 'mouth',
    11: 'upper_lip',
    12: 'lower_lip',
    13: 'hair',
    14: 'hat',
    15: 'earring',
    16: 'necklace',
    17: 'neck',
    18: 'clothing'
}

# Face Mask에 포함되는 클래스
DEFAULT_FACE_LABELS = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # 얼굴
DEFAULT_HAIR_LABELS = [13]  # 머리카락
```

#### Face Masking 처리 흐름

```
[Style Image (1024x1024)]
        |
        v
+-------------------+
| SegFormer         |
| Segmentation      |
+--------+----------+
         |
         v
[19-class Segmentation Map]
         |
         v
+-------------------+
| Select Face+Hair  |
| Labels            |
+--------+----------+
         |
         v
[Binary Mask (Face=White)]
         |
         +-- expand_pixels (기본 10px)
         |
         +-- edge_blur (기본 10px)
         |
         v
[Soft Edge Mask]
         |
         v
+-------------------+
| Apply Masking     |
| Method            |
+--------+----------+
         |
    +----+----+
    |    |    |
    v    v    v
[Blur][Fill][Noise]
    |
    v
[Masked Style Image]
```

**마스킹 방법**:

| Method | 설명 | 효과 |
|--------|------|------|
| `gaussian_blur` | 얼굴 영역 강한 blur (기본) | 얼굴 특징 제거, 색감 유지 |
| `fill` | 회색(128,128,128)으로 채움 | 완전 중립화 |
| `noise` | 가우시안 노이즈로 채움 | 랜덤한 텍스처 |

### 5. Auxiliary Models

| Model | 용도 | 로드 시점 |
|-------|------|----------|
| MiDaS | Depth Map 추출 | ControlNet 사용시 lazy load |
| CLIP ViT-H-14 | 이미지 임베딩 | Pipeline 초기화시 |
| VAE (AutoencoderKL) | Latent 인코딩/디코딩 | Pipeline 초기화시 |
| TinyVAE (Optional) | 빠른 디코딩 | 선택적 사용 |
| Text Encoders | 프롬프트 인코딩 | Pipeline 초기화시 |

---

## Key Algorithms

### 1. Decoupled Classifier-Free Guidance (DCG)

**문제**: 표준 CFG는 모든 조건을 동일하게 처리
**해결**: 텍스트/이미지 guidance를 분리하여 개별 제어

#### DCG Types

```
Standard CFG:
noise = uncond + w * (cond - uncond)

DCG Type 3 (기본):
batch = [uncond, text_cond, image_cond]
noise = uncond + a*(text - uncond) + b*(image - uncond)

DCG Type 4:
batch = [uncond, text_cond, image_cond, combined_cond]
noise = uncond + a*(text - uncond) + b*(image - uncond)
```

#### 스케줄러

DCG는 timestep별로 다른 guidance scale 적용:

```python
# configs/fastface/am2_and_dcg.json 예시
"sch_kwargs": {
    "custom": {
        "a": [1.0, 1.5, 1.5, 1.0],   # timestep별 text guidance
        "b": [1.0, 3.0, 3.0, 1.0]    # timestep별 image guidance
    }
}
```

### 2. CLIP Blending (v4+)

**문제**: 스타일 이미지에 인물이 없으면 identity loss 발생
**원인**: Style CLIP이 "사람 없음" semantic을 인코딩

#### 해결 방법

```python
# Face CLIP: "사람이 있는" semantic 보존
face_crop = align_face(face_image)
face_clip = clip_encoder(face_crop)  # [1, 257, 1280]

# Style CLIP: 스타일 정보
style_clip = clip_encoder(style_image)  # [1, 257, 1280]

# Blending: 사람 존재 + 스타일 조합
blended = (1 - style_strength) * face_clip + style_strength * style_clip
```

| style_strength | Face 비중 | Style 비중 | 결과 |
|----------------|----------|-----------|------|
| 0.0 | 100% | 0% | 순수 얼굴 |
| 0.3 (기본) | 70% | 30% | 균형 |
| 0.5 | 50% | 50% | 강한 스타일 |
| 0.7+ | 30%- | 70%+ | identity 약화 위험 |

### 3. Face Masking Algorithm (v6+)

**문제**: 스타일 이미지에 다른 사람 얼굴이 있으면:
- FaceID: "이 얼굴로 생성해"
- Style CLIP/Latents/ControlNet: "저 얼굴로 생성해"
- 결과: 두 얼굴 특징이 섞여 어색한 결과

**해결**: 스타일 이미지의 얼굴 영역을 제거

```
Before v6:
[Style Face] + [Target FaceID] --> [Confused Mix]

After v6:
[Blurred Style Face] + [Target FaceID] --> [Clean Target Face]
```

#### v7 Aspect Ratio Adjustment

입력 얼굴과 스타일 얼굴의 비율이 다를 때 마스크 조정:

```python
# 입력 얼굴 비율 계산
input_face_aspect = input_face_height / input_face_width

# 스타일 마스크를 입력 얼굴 비율에 맞춤
if input_face_aspect > style_mask_aspect:
    # 입력이 더 길쭉함 -> 마스크 세로 확장
    expand_mask_vertically(scale=input_face_aspect/style_mask_aspect)
else:
    # 입력이 더 넓음 -> 마스크 가로 확장
    expand_mask_horizontally(scale=style_mask_aspect/input_face_aspect)
```

### 4. Attention Manipulation (Custom DCA)

UNet의 attention layer에 직접 개입하여 생성 제어:

```python
# src/custom_dca.py
target_parts = ["down", "up"]      # UNet 블록
target_tokens = [0, 1, 2, 3]       # Attention 토큰
target_tsteps = [0, 1, 2, 3]       # Denoising 초기 단계

# 적용 가능한 변환
am_transforms = [
    "adaptive_softmask",  # 적응형 sigmoid (quantile 기반)
    "scale",              # 선형 스케일링
    "pow",                # 거듭제곱 변환
    "softmask"            # 고정 sigmoid
]
```

---

## Data Flow

### 1. Generation Request Flow

```
Frontend                         Backend                          Pipeline
   |                               |                                 |
   | POST /generate                |                                 |
   |------------------------------>|                                 |
   |                               | Create Task (pending)           |
   |                               | Start Background Thread         |
   |                               |-------------------------------->|
   |                               |                                 |
   | GET /tasks/{id}               |     Load Models (if needed)     |
   |------------------------------>|                                 |
   | {status: "running"}           |     Phase 1: Face Processing    |
   |<------------------------------|     - InsightFace detection     |
   |                               |     - FaceID embedding          |
   |                               |     - Face aspect ratio         |
   |                               |                                 |
   | GET /tasks/{id}               |     Phase 2: Style Processing   |
   |------------------------------>|     - Face detection            |
   | {status: "running",           |     - SegFormer parsing         |
   |  progress: "Extracting..."}   |     - Face masking              |
   |<------------------------------|     - Save masked preview       |
   |                               |                                 |
   |                               |     Phase 3: ControlNet         |
   |                               |     - MiDaS depth               |
   |                               |     - Mask depth face           |
   |                               |                                 |
   |                               |     Phase 4: CLIP               |
   |                               |     - Face CLIP                 |
   |                               |     - Style CLIP                |
   |                               |     - Blending                  |
   |                               |                                 |
   |                               |     Phase 5: Generation         |
   |                               |     - SDXL + IP-Adapter + DCG   |
   |                               |                                 |
   |                               |<--------------------------------|
   |                               |     Return result               |
   |                               |                                 |
   |                               | Save output                     |
   |                               | Create history                  |
   |                               | Update task (completed)         |
   |                               |                                 |
   | GET /tasks/{id}               |                                 |
   |------------------------------>|                                 |
   | {status: "completed",         |                                 |
   |  image_url: "/output/...",    |                                 |
   |  masked_style_url: "..."}     |                                 |
   |<------------------------------|                                 |
```

### 2. Mask Preview Flow (v7)

```
Frontend                         Backend
   |                               |
   | POST /preview-mask            |
   | {image_id, image_type}        |
   |------------------------------>|
   |                               |
   |                               | if image_type == "face":
   |                               |   InsightFace -> BBox overlay
   |                               |
   |                               | if image_type == "style":
   |                               |   SegFormer -> Face mask
   |                               |   Apply blur method
   |                               |   Create overlay image
   |                               |
   | {success, overlay_url,        |
   |  mask_coverage, bbox,         |
   |  aspect_ratio}                |
   |<------------------------------|
```

---

## Parameter Reference

### Generation Parameters

| Parameter | Type | Default | Range | 설명 |
|-----------|------|---------|-------|------|
| `ips` | float | 0.8 | 0.0-1.5 | IP-Adapter FaceID 강도 |
| `lora_scale` | float | 0.6 | 0.0-1.0 | LoRA 강도 (Hyper 모델) |
| `style_strength` | float | 0.3 | 0.0-1.0 | CLIP 블렌딩 스타일 비율 |
| `denoising_strength` | float | 0.6 | 0.2-1.0 | img2img denoising |
| `inference_steps` | int | 4 | 1-50 | 생성 스텝 수 |
| `seed` | int | 42 | any | 랜덤 시드 |

### v6 Face Masking Parameters

| Parameter | Type | Default | 설명 |
|-----------|------|---------|------|
| `mask_style_face` | bool | true | 스타일 얼굴 마스킹 활성화 |
| `face_mask_method` | string | "gaussian_blur" | 마스킹 방법 |
| `include_hair_in_mask` | bool | true | 머리카락 포함 여부 |
| `face_mask_blur_radius` | int | 50 | blur 강도 (px) |

### v7 Advanced Parameters

| Parameter | Type | Default | Range | 설명 |
|-----------|------|---------|-------|------|
| `mask_expand_pixels` | int | 10 | 0-50 | 마스크 확장 (px) |
| `mask_edge_blur` | int | 10 | 0-30 | 마스크 경계 blur |
| `controlnet_scale` | float | 0.4 | 0.0-1.0 | 구조 보존 강도 |
| `depth_blur_radius` | int | 80 | 0-150 | depth map 얼굴 blur |
| `style_strength_cap` | float | 0.10 | 0.0-0.5 | 마스킹시 스타일 상한 |
| `denoising_min` | float | 0.90 | 0.5-1.0 | 마스킹시 denoising 하한 |

**v7 파라미터 동작**:

Face Masking이 활성화되면 (`mask_style_face=true` AND 스타일에 얼굴 감지):
- `style_strength` -> `style_strength_cap` 사용 (스타일 영향 제한)
- `denoising_strength` -> `denoising_min` 사용 (더 자유로운 생성)

이유: 마스킹된 얼굴 영역은 FaceID가 완전히 새로 생성해야 하므로 높은 denoising 필요

---

## API Reference

### POST /generate

이미지 생성 요청 (비동기)

```typescript
Request:
{
  image_id: string           // 업로드된 얼굴 이미지 ID
  prompt: string             // 텍스트 프롬프트
  negative_prompt?: string   // 네거티브 프롬프트
  model_name: string         // 모델 선택 (hyper/realvis/...)
  inference_steps: number    // 생성 스텝
  ips: number                // IP-Adapter 강도
  lora_scale: number         // LoRA 강도
  seed: number               // 시드
  style_image_id?: string    // 스타일 이미지 ID
  style_strength?: number    // CLIP 블렌딩 비율
  denoising_strength?: number // img2img 강도
  // v6 params
  mask_style_face?: boolean
  face_mask_method?: string
  include_hair_in_mask?: boolean
  face_mask_blur_radius?: number
  // v7 params
  mask_expand_pixels?: number
  mask_edge_blur?: number
  controlnet_scale?: number
  depth_blur_radius?: number
  style_strength_cap?: number
  denoising_min?: number
  title?: string
  use_tiny_vae?: boolean
}

Response:
{
  success: boolean
  task_id: string
  status: "pending"
}
```

### GET /tasks/{task_id}

Task 상태 조회

```typescript
Response:
{
  id: string
  status: "pending" | "running" | "completed" | "failed"
  created_at: string
  updated_at: string
  input_image_url?: string
  style_image_url?: string
  masked_style_image_url?: string  // v7: 마스킹된 스타일 미리보기
  image_url?: string               // 생성 결과
  error?: string
  history_id?: string
  params: GenerateParams
  progress_message?: string
}
```

### POST /preview-mask

마스크 미리보기 생성

```typescript
Request:
{
  image_id: string
  image_type: "face" | "style"
  include_hair_in_mask?: boolean
  mask_expand_pixels?: number
  mask_edge_blur?: number
  face_mask_method?: string
  face_mask_blur_radius?: number
}

Response:
{
  success: boolean
  overlay_url?: string      // 오버레이 이미지 URL
  error?: string
  face_detected?: boolean   // face type only
  bbox?: number[]           // [x1, y1, x2, y2]
  aspect_ratio?: number     // height/width
  mask_coverage?: number    // style type only (%)
}
```

### POST /generate-prompt

VLM 기반 프롬프트 자동 생성 (Gemini 2.5 Flash)

```typescript
Request:
{
  face_image_id: string
  style_image_id?: string
}

Response:
{
  success: boolean
  positive: string    // 긍정 프롬프트
  negative: string    // 네거티브 프롬프트
}
```

---

## File Structure

```
FastFace/
+-- src/
|   +-- sdxl_custom_pipeline.py   # 핵심 파이프라인 (execute 메서드)
|   +-- face_parsing.py           # SegFormer 기반 얼굴 파싱 (v6+)
|   +-- custom_dca.py             # Attention Manipulation
|   +-- dcg.py                    # Decoupled Guidance 구현
|   +-- cfg_schedulers.py         # CFG 스케줄러
|   +-- controlnet_pipeline.py    # ControlNet 유틸리티
|   +-- utils.py                  # 헬퍼 함수
|
+-- backend/
|   +-- main.py                   # FastAPI 서버 (API 엔드포인트)
|
+-- frontend/
|   +-- app/
|   |   +-- layout.tsx            # Root 레이아웃
|   |   +-- page.tsx              # 메인 UI
|   +-- components/
|   |   +-- Icons.tsx             # 아이콘 컴포넌트
|   |   +-- Sidebar.tsx           # 사이드바
|   +-- lib/
|       +-- api.ts                # API 클라이언트
|
+-- configs/
|   +-- fastface/
|       +-- am1_and_dcg.json      # DCG Type 3 설정
|       +-- am2_and_dcg.json      # DCG Type 3 + 커스텀 스케줄
|
+-- models_cache/                  # 모델 캐시 디렉토리
+-- data/
|   +-- uploads/                   # 업로드된 이미지
|   +-- output/                    # 생성된 이미지
|   +-- history.json              # 히스토리 데이터
|   +-- folders.json              # 폴더 데이터
|
+-- docs/
    +-- V5_ARCHITECTURE.md        # v5 아키텍처 문서
    +-- V7_ARCHITECTURE.md        # 이 문서
    +-- DEVELOPMENT_HISTORY.md    # 개발 히스토리
```

---

## Memory Requirements

| Component | VRAM (fp16) | Notes |
|-----------|-------------|-------|
| SDXL UNet | ~6GB | 핵심 |
| VAE | ~1GB | AutoencoderKL |
| IP-Adapter FaceID | ~0.5GB | Plus v2 |
| ControlNet Depth | ~2GB | Optional |
| SegFormer | ~0.1GB | Lazy load |
| MiDaS | ~0.5GB | Lazy load |
| **Total (기본)** | ~8GB | ControlNet 미사용시 |
| **Total (전체)** | ~10GB | ControlNet 사용시 |

### Platform Support

| Platform | dtype | Notes |
|----------|-------|-------|
| CUDA | float16 | 권장, 최고 성능 |
| MPS (Mac) | float16 | M1/M2/M3 지원 |
| CPU | float32 | 매우 느림 |

---

## Troubleshooting

### 문제: Identity가 약하게 반영됨

```
원인: ips (ip_adapter_scale)가 낮음
해결: ips를 0.8-1.0으로 높임
```

### 문제: 스타일이 반영되지 않음

```
원인: style_strength가 너무 낮거나, mask_style_face로 인해 cap 적용됨
해결:
- style_strength 0.3-0.5로 조정
- style_strength_cap 값 확인 (기본 0.10)
- 스타일 이미지에 얼굴이 없는지 확인
```

### 문제: 두 얼굴이 섞인 것 같은 결과

```
원인: 스타일 이미지 얼굴이 제대로 마스킹되지 않음
해결:
- mask_style_face = true 확인
- face_mask_blur_radius 높임 (50-80)
- mask_expand_pixels 높임 (10-30)
```

### 문제: MPS dtype mismatch 에러

```
Error: 'mps.add' op requires the same element type
원인: ControlNet과 pipeline의 dtype 불일치
해결: ControlNet을 float16으로 통일 (v5에서 수정됨)
```

### 문제: 얼굴이 검출되지 않음

```
Error: No face detected in the input image
원인: 얼굴이 너무 작거나 측면/가려짐
해결:
- 정면 얼굴 이미지 사용
- 최소 112x112 이상 해상도
- 얼굴이 이미지의 30% 이상 차지하도록
```

### 문제: 생성이 너무 느림

```
원인: 모델 로딩 또는 스텝 수
해결:
- Hyper 모델 사용 (4-8 steps)
- use_tiny_vae = true
- inference_steps 줄임
```

---

## Configuration Examples

### 기본 설정 (권장)

```json
{
  "ips": 0.8,
  "lora_scale": 0.6,
  "style_strength": 0.3,
  "denoising_strength": 0.6,
  "inference_steps": 4,
  "mask_style_face": true,
  "face_mask_method": "gaussian_blur",
  "include_hair_in_mask": true,
  "face_mask_blur_radius": 50,
  "mask_expand_pixels": 10,
  "mask_edge_blur": 10,
  "controlnet_scale": 0.4,
  "depth_blur_radius": 80,
  "style_strength_cap": 0.10,
  "denoising_min": 0.90
}
```

### 강한 얼굴 유지

```json
{
  "ips": 1.0,
  "style_strength": 0.2,
  "style_strength_cap": 0.05,
  "denoising_min": 0.95
}
```

### 강한 스타일 적용 (마스킹 없는 스타일 이미지용)

```json
{
  "ips": 0.7,
  "style_strength": 0.5,
  "denoising_strength": 0.5,
  "mask_style_face": false
}
```

### 포즈만 참조

```json
{
  "ips": 0.9,
  "style_strength": 0.1,
  "controlnet_scale": 0.6,
  "mask_style_face": true,
  "style_strength_cap": 0.05
}
```

---

## References

- IP-Adapter: https://github.com/tencent-ailab/IP-Adapter
- ControlNet: https://github.com/lllyasviel/ControlNet
- InsightFace: https://github.com/deepinsight/insightface
- SegFormer: https://huggingface.co/jonathandinu/face-parsing
- Hyper-SD: https://huggingface.co/ByteDance/Hyper-SD
- RealVisXL: https://huggingface.co/SG161222/RealVisXL_V5.0

---

## Changelog

### v7 (Current)
- 프론트엔드에서 모든 마스킹 파라미터 제어 가능
- 백엔드 자동 오버라이드 제거
- 마스크 미리보기 기능 (실제 생성시 사용된 마스크 표시)
- 입력 얼굴-스타일 얼굴 비율 자동 조정

### v6
- Face Masking 도입 (스타일 이미지 얼굴 충돌 해결)
- SegFormer 기반 얼굴+머리카락 세그멘테이션
- Depth Map 얼굴 영역 마스킹

### v5
- ControlNet Depth 통합
- 스타일 이미지 구조 보존

### v4
- CLIP Blending 도입
- Identity Loss 문제 해결

### v3
- Batch-Wise DCG
- 세밀한 guidance 제어

### v2
- RealVisXL 기본 모델
- 비동기 Task 시스템
- 히스토리/폴더 기능

### v1
- Mac MPS 지원
- Web UI 도입
