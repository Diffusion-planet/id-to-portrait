<p align="center">
  <img src="images/logo.svg" alt="Prometheus Logo" width="300">
</p>

<h1 align="center">ID to Portrait</h1>

<p align="center">
  <strong>얼굴 정체성 보존 스타일 트랜스퍼 시스템</strong>
</p>

<p align="center">
  <a href="#소개">소개</a> |
  <a href="#주요-기능">주요 기능</a> |
  <a href="#설치">설치</a> |
  <a href="#사용법">사용법</a> |
  <a href="#아키텍처">아키텍처</a> |
  <a href="#팀원">팀원</a>
</p>

---

## 소개

ID to Portrait는 사용자의 얼굴 이미지에서 정체성(identity)을 보존하면서, 스타일 이미지의 배경, 포즈, 색감을 반영한 새로운 초상화를 생성하는 AI 시스템이다.

IP-Adapter FaceID와 SDXL 기반의 Diffusion 모델을 활용하여 빠르고 고품질의 얼굴 생성을 지원한다.

```
[얼굴 이미지] + [스타일 이미지] + [텍스트 프롬프트] --> [생성된 초상화]
     |               |                   |
  정체성 보존     포즈/배경/색감      추가 수정 지시
```

### 프로젝트 정보

- 프로메테우스 2025년 2학기 데모데이 프로젝트 4팀
- [FastFace 논문](https://arxiv.org/abs/2505.21144) 기반 확장 개발

---

## 주요 기능

### 핵심 기능

- **얼굴 정체성 보존**: IP-Adapter FaceID를 통한 얼굴 특징 추출 및 보존
- **스타일 트랜스퍼**: 스타일 이미지의 배경, 포즈, 색감 반영
- **Face Masking**: 스타일 이미지 내 얼굴 자동 감지 및 마스킹으로 정체성 충돌 방지
- **ControlNet Depth**: 스타일 이미지의 공간 구조 보존
- **CLIP Blending**: Face/Style CLIP 임베딩 블렌딩으로 identity loss 해결

### 지원 모델

| 모델 | 설명 | 스텝 수 |
|------|------|---------|
| Hyper-SD | 고속 생성 (권장) | 4-8 |
| RealVisXL | 고품질 실사 | 20-50 |
| SDXL-Lightning | 빠른 생성 | 2-8 |
| LCM-LoRA | 일관성 모델 | 4-8 |
| SDXL-Turbo | 초고속 | 1-4 |

### 플랫폼 지원

| 플랫폼 | 지원 여부 | 비고 |
|--------|----------|------|
| CUDA (NVIDIA) | O | 권장, 최고 성능 |
| MPS (Mac Apple Silicon) | O | M1/M2/M3 지원 |
| CPU | O | 매우 느림 |

---

## 설치

Python 3.10 또는 3.11 권장

### CUDA (Linux/Windows with NVIDIA GPU)

```bash
# 저장소 클론
git clone https://github.com/Diffusion-planet/id-to-portrait.git
cd id-to-portrait

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Windows: venv\Scripts\activate

# PyTorch CUDA 설치 (cu121은 CUDA 버전에 맞게 조정)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 의존성 설치
pip install -r requirements_cuda.txt
```

### Mac MPS (Apple Silicon)

```bash
# 자동 설치
chmod +x setup_mac.sh
./setup_mac.sh

# 또는 수동 설치
python -m venv venv_mps
source venv_mps/bin/activate
pip install -r requirements_mps.txt
```

### CPU Only

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_cpu.txt
```

### 환경 설정

```bash
cp backend/.env.example backend/.env
```

주요 설정:
- `DEVICE`: `auto` (권장), `cuda`, `mps`, `cpu`
- `GEMINI_API_KEY`: VLM 프롬프트 자동 생성용 (선택)

---

## 사용법

### Web UI 실행

**백엔드 실행:**

```bash
source venv_mps/bin/activate  # 또는 사용 중인 가상환경
python -m backend.main
```

백엔드: `http://localhost:8000`

**프론트엔드 실행:**

```bash
cd frontend
npm install
npm run dev
```

프론트엔드: `http://localhost:3000`

### 주요 파라미터

| 파라미터 | 기본값 | 범위 | 설명 |
|----------|--------|------|------|
| `ips` | 0.8 | 0.0-1.5 | IP-Adapter FaceID 강도 |
| `style_strength` | 0.3 | 0.0-1.0 | CLIP 블렌딩 스타일 비율 |
| `denoising_strength` | 0.6 | 0.2-1.0 | img2img denoising |
| `inference_steps` | 4 | 1-50 | 생성 스텝 수 |
| `controlnet_scale` | 0.4 | 0.0-1.0 | 구조 보존 강도 |

### Face Masking 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `mask_style_face` | true | 스타일 얼굴 마스킹 활성화 |
| `face_mask_method` | gaussian_blur | 마스킹 방법 |
| `include_hair_in_mask` | true | 머리카락 포함 여부 |
| `face_mask_blur_radius` | 50 | blur 강도 (px) |

---

## 아키텍처

### 시스템 구조

```
Frontend (Next.js)          Backend (FastAPI)              Pipeline (PyTorch)
      |                           |                              |
      | POST /generate            |                              |
      |-------------------------->|                              |
      |                           | Create Task                  |
      |                           |----------------------------->|
      |                           |                              |
      | GET /tasks/{id}           |     1. Face Processing       |
      |-------------------------->|        - InsightFace         |
      | {status: "running"}       |        - FaceID embedding    |
      |<--------------------------|                              |
      |                           |     2. Style Processing      |
      |                           |        - Face detection      |
      |                           |        - SegFormer parsing   |
      |                           |        - Face masking        |
      |                           |                              |
      |                           |     3. ControlNet Depth      |
      |                           |        - MiDaS depth         |
      |                           |                              |
      |                           |     4. CLIP Blending         |
      |                           |                              |
      |                           |     5. SDXL Generation       |
      |                           |        + IP-Adapter + DCG    |
      |                           |                              |
      | GET /tasks/{id}           |<-----------------------------|
      |-------------------------->|                              |
      | {status: "completed",     |                              |
      |  image_url: "..."}        |                              |
      |<--------------------------|                              |
```

### 핵심 모델

1. **SDXL (Stable Diffusion XL)**: 기본 생성 모델
2. **IP-Adapter FaceID Plus v2**: 얼굴 정체성 주입
3. **ControlNet Depth**: 공간 구조 보존
4. **SegFormer**: 얼굴/머리카락 세그멘테이션
5. **InsightFace**: 얼굴 검출 및 임베딩 추출

### 디렉토리 구조

```
id-to-portrait/
|-- src/
|   |-- sdxl_custom_pipeline.py   # 핵심 파이프라인
|   |-- face_parsing.py           # 얼굴 파싱
|   |-- custom_dca.py             # Attention Manipulation
|   |-- dcg.py                    # Decoupled Guidance
|   |-- controlnet_pipeline.py    # ControlNet 유틸
|
|-- backend/
|   |-- main.py                   # FastAPI 서버
|
|-- frontend/
|   |-- app/                      # Next.js 앱
|   |-- components/               # React 컴포넌트
|   |-- lib/                      # API 클라이언트
|
|-- configs/
|   |-- fastface/                 # DCG 설정
|
|-- docs/                         # 문서
|-- models_cache/                 # 모델 캐시
|-- data/                         # 업로드/출력 이미지
```

---

## 시스템 요구사항

### GPU 메모리

| 구성요소 | VRAM (fp16) |
|----------|-------------|
| SDXL UNet | ~6GB |
| VAE | ~1GB |
| IP-Adapter FaceID | ~0.5GB |
| ControlNet Depth | ~2GB |
| SegFormer | ~0.1GB |
| **총합 (ControlNet 포함)** | ~10GB |

### 권장 사양

- NVIDIA GPU: 10GB+ VRAM (RTX 3080 이상 권장)
- Mac: M1/M2/M3 (16GB+ 통합 메모리 권장)
- RAM: 16GB+
- 저장공간: 20GB+ (모델 캐시용)

---

## 팀원

| 임병건 (Byungkun Lim) | 이성민 (Seongmin Lee) |
| :---: | :---: |
| <img src="https://avatars.githubusercontent.com/byungkun0823" width="150px" alt="Byungkun Lim" /> | <img src="https://avatars.githubusercontent.com/danlee-dev" width="150px" alt="Seongmin Lee" /> |
| [GitHub: @byungkun0823](https://github.com/byungkun0823) | [GitHub: @danlee-dev](https://github.com/danlee-dev) |
| 고려대학교 컴퓨터학과 | 고려대학교 컴퓨터학과 |

| 최서연 (Seoyeon Choi) | 홍지연 (Jiyeon Hong) |
| :---: | :---: |
| <img src="https://avatars.githubusercontent.com/seoyeon-eo" width="150px" alt="Seoyeon Choi" /> | <img src="https://avatars.githubusercontent.com/hongjiyeon56" width="150px" alt="Jiyeon Hong" /> |
| [GitHub: @seoyeon-eo](https://github.com/seoyeon-eo) | [GitHub: @hongjiyeon56](https://github.com/hongjiyeon56) |
| 고려대학교 컴퓨터학과 | 고려대학교 컴퓨터학과 |

---

## 참고 자료

- [FastFace Paper](https://arxiv.org/abs/2505.21144)
- [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter)
- [ControlNet](https://github.com/lllyasviel/ControlNet)
- [InsightFace](https://github.com/deepinsight/insightface)
- [SegFormer Face Parsing](https://huggingface.co/jonathandinu/face-parsing)
- [Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD)
- [RealVisXL](https://huggingface.co/SG161222/RealVisXL_V5.0)

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따른다.

---

## Citation

이 프로젝트는 FastFace 논문을 기반으로 한다:

```bibtex
@misc{karpukhin2025fastfacetuningidentitypreservation,
      title={FastFace: Tuning Identity Preservation in Distilled Diffusion via Guidance and Attention},
      author={Sergey Karpukhin and Vadim Titov and Andrey Kuznetsov and Aibek Alanov},
      year={2025},
      eprint={2505.21144},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.21144},
}
```
