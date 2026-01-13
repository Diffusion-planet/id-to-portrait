# FastFace v6 Architecture Documentation

## Overview

FastFace v6는 **스타일 이미지의 얼굴 마스킹** 기능을 추가하여 스타일 이미지에 있는 얼굴과 FaceID가 생성하려는 얼굴 간의 충돌을 해결합니다.

### v6 핵심 목표

```
[Face Image] + [Style Image (with face)] + [Text Prompt]
                     |
                     v
              [Face Masked Style Image]
                     |
                     v
              [Generated Image - No Face Conflict]
```

---

## Problem Statement (v5 Issue #3)

### 스타일 이미지에 얼굴이 있을 때 충돌 발생

**문제 상황:**
- 스타일 이미지: 다른 사람의 얼굴이 포함된 사진
- FaceID: 입력된 얼굴 이미지의 정체성을 생성하려고 함
- 결과: 얼굴 형태가 뭉개지거나, 스타일 얼굴과 FaceID 얼굴이 혼합됨

**충돌 구조:**

```
Style Image (with face)
    |
    +---> img2img latents: 스타일 이미지 얼굴 구조 인코딩
    +---> ControlNet depth: 스타일 이미지 얼굴 깊이 유지
    +---> Style CLIP: 스타일 이미지 얼굴 semantic 포함
    +---> FaceID: 다른 얼굴 생성 시도
          |
          +---> 충돌! 4개 신호 중 3개가 "스타일 얼굴 유지"
```

**신호 충돌 테이블:**

| Component | 스타일 얼굴 없음 | 스타일 얼굴 있음 (v5) | 스타일 얼굴 있음 (v6) |
|-----------|---------------|-------------------|-------------------|
| img2img latents | FaceID 자유 | 스타일 얼굴 유지 | 마스킹됨 (blur) |
| ControlNet depth | FaceID 자유 | 스타일 얼굴 깊이 | 마스킹됨 (neutral) |
| Style CLIP | 스타일 전달 | 스타일 얼굴 포함 | 스타일 얼굴 blur |
| FaceID | 자유 생성 | 충돌 | 자유 생성 |
| **결과** | 성공 | 혼합/왜곡 | 성공 |

---

## v6 Solution: Face Masking

### 전체 구조도

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastFace v6 Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐                                                   │
│  │ Style Image  │                                                   │
│  └──────┬───────┘                                                   │
│         │                                                           │
│         ├──► InsightFace ──► Face Detected?                         │
│         │                         │                                 │
│         │                    Yes  │  No                             │
│         │         ┌───────────────┴───────────────┐                 │
│         │         ▼                               ▼                 │
│         │  ┌──────────────┐                Use Original             │
│         │  │  SegFormer   │                Style Image              │
│         │  │ Face Parser  │                       │                 │
│         │  └──────┬───────┘                       │                 │
│         │         │                               │                 │
│         │         ▼                               │                 │
│         │  ┌──────────────┐                       │                 │
│         │  │  Face+Hair   │                       │                 │
│         │  │    Mask      │                       │                 │
│         │  └──────┬───────┘                       │                 │
│         │         │                               │                 │
│         │         ├────────────┬─────────────┐    │                 │
│         │         ▼            ▼             ▼    │                 │
│         │  ┌──────────┐ ┌──────────┐ ┌──────────┐ │                 │
│         │  │  Apply   │ │  Apply   │ │ (Style   │ │                 │
│         │  │  Blur    │ │  to Depth│ │   CLIP)  │ │                 │
│         │  │  Mask    │ │  Map     │ │ Uses blur│ │                 │
│         │  └────┬─────┘ └────┬─────┘ └────┬─────┘ │                 │
│         │       │            │            │       │                 │
│         │       ▼            ▼            ▼       ▼                 │
│         │  ┌──────────────────────────────────────────┐             │
│         │  │         Masked Style Processing          │             │
│         │  │                                          │             │
│         │  │  - VAE Encode (from masked image)        │             │
│         │  │  - Depth Extract (from masked image)     │             │
│         │  │  - CLIP Embed (from masked image)        │             │
│         │  └────────────────────┬─────────────────────┘             │
│         │                       │                                   │
│         │                       ▼                                   │
│         │               [Rest of v5 Pipeline]                       │
│         │               (FaceID + DCG + ControlNet)                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. FaceParser Module (`src/face_parsing.py`)

모듈화된 얼굴 파싱 클래스로, 깔끔한 코드 구조와 재사용성을 제공합니다.

```python
from src.face_parsing import FaceParser

parser = FaceParser(device="mps")
mask = parser.get_face_hair_mask(image)
masked_image = parser.apply_mask(image, mask, method='gaussian_blur')
```

#### SegFormer Face Parsing Labels

| Label | Name | Mask 포함 |
|-------|------|----------|
| 0 | background | No |
| 1 | skin | Yes |
| 2 | nose | Yes |
| 3 | eyeglasses | No |
| 4 | left_eye | Yes |
| 5 | right_eye | Yes |
| 6 | left_eyebrow | Yes |
| 7 | right_eyebrow | Yes |
| 8 | left_ear | Yes |
| 9 | right_ear | Yes |
| 10 | mouth | Yes |
| 11 | upper_lip | Yes |
| 12 | lower_lip | Yes |
| 13 | hair | Yes (optional) |
| 14-18 | accessories/clothing | No |

### 2. Mask Application Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `gaussian_blur` | 얼굴 영역을 강하게 블러 | 기본값, 가장 자연스러움 |
| `fill` | 단색으로 채움 (gray) | 디버깅, 실험용 |
| `noise` | 가우시안 노이즈로 채움 | 대안 실험용 |

### 3. Depth Map Masking

ControlNet이 스타일 얼굴의 깊이 정보를 강제하지 않도록, 마스크된 영역의 깊이를 중립값(0.5)으로 설정합니다.

```python
# Before v6: depth_image has style face depth
# After v6: depth_image has neutral depth in face region
depth_image = face_parser.apply_mask_to_depth(depth_image, face_mask, fill_value=0.5)
```

---

## API Parameters

### New v6 Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mask_style_face` | bool | True | 스타일 이미지 얼굴 마스킹 활성화 |
| `face_mask_method` | str | "gaussian_blur" | 마스킹 방법 ("gaussian_blur", "fill", "noise") |
| `include_hair_in_mask` | bool | True | 머리카락도 마스킹에 포함 |
| `face_mask_blur_radius` | int | 50 | gaussian_blur 방법의 블러 강도 |

### API Usage

```bash
curl -X POST http://localhost:8007/generate \
  -F "image_id=abc123" \
  -F "prompt=professional photo" \
  -F "style_image_id=xyz789" \
  -F "mask_style_face=true" \
  -F "face_mask_method=gaussian_blur" \
  -F "include_hair_in_mask=true" \
  -F "face_mask_blur_radius=50"
```

---

## Execution Flow

### v6 Face Masking Flow

```
1. Style Image Loaded
         │
         ▼
2. InsightFace Detection
         │
    ┌────┴────┐
    │ Face?   │
    └────┬────┘
    Yes  │  No
    ▼    │    ▼
3. FaceParser.get_face_hair_mask()  │  Skip Masking
         │                          │
         ▼                          │
4. FaceParser.apply_mask()          │
   (blur face region)               │
         │                          │
         ├──────────────────────────┘
         ▼
5. Use Masked Image for:
   - VAE Encoding (init_latents)
   - Depth Extraction (ControlNet)
   - Style CLIP Embedding
         │
         ▼
6. FaceParser.apply_mask_to_depth()
   (neutralize face depth)
         │
         ▼
7. Continue with v5 Pipeline
   (FaceID freely generates face)
```

---

## File Structure

```
FastFace/
├── src/
│   ├── sdxl_custom_pipeline.py    # v6: face_parser property 추가
│   ├── face_parsing.py            # NEW: 모듈화된 FaceParser 클래스
│   ├── dcg.py                     # Decoupled Guidance
│   └── utils.py                   # 헬퍼 함수
├── backend/
│   └── main.py                    # v6: face masking params 추가
├── tests/
│   └── test_face_masking.py       # NEW: 얼굴 마스킹 유닛 테스트
└── docs/
    ├── V5_ARCHITECTURE.md
    └── V6_ARCHITECTURE.md         # 이 문서
```

---

## Memory Requirements

| Component | VRAM Usage | Notes |
|-----------|-----------|-------|
| SDXL UNet | ~6GB | float16 |
| VAE | ~1GB | float16 |
| IP-Adapter FaceID | ~0.5GB | |
| ControlNet Depth | ~2GB | Optional |
| MiDaS Depth | ~0.5GB | CPU/lazy loaded |
| **SegFormer Face Parser** | **~0.1GB** | **NEW (v6)** |
| **Total** | ~10GB | with ControlNet |

---

## Testing

### Unit Test

```bash
source venv_mps/bin/activate
python tests/test_face_masking.py --image path/to/face.jpg
```

### Test Output

```
test_output/
├── original.png      # 원본 이미지
├── mask.png          # 추출된 얼굴+머리카락 마스크
├── masked_blur.png   # gaussian_blur 적용
├── masked_fill.png   # fill 적용 (gray)
└── masked_noise.png  # noise 적용
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v0 | 2025-05-28 | 논문 원본 구현 |
| v1 | 2025-12-05 | Mac MPS 지원, Web UI |
| v2 | 2025-12-08 | RealVisXL, 비동기 Task |
| v3 | 2025-12-08 | Batch-Wise DCG |
| v4 | 2025-12-11 | CLIP Blending (Identity Loss 해결) |
| v5 | 2025-12-11 | ControlNet 통합, 구조 보존 |
| **v6** | **2025-01-13** | **Face Masking (스타일 얼굴 충돌 해결)** |

---

## References

- [SegFormer Face Parsing (HuggingFace)](https://huggingface.co/jonathandinu/face-parsing)
- [Face Swap via Diffusion Model](https://arxiv.org/abs/2403.01108)
- [Realistic Face Swapping](https://arxiv.org/abs/2409.07269)

---

## Related Issues

- Issue #1 - Identity Loss 문제 (해결됨, v4)
- Issue #2 - 인물 없는 스타일 이미지 문제 (진행중)
- **Issue #3 - 스타일 이미지 얼굴 충돌 문제 (해결됨, v6)**
