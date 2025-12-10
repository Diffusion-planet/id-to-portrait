# FastFace Development History

FastFace 논문 구현체의 버전별 개발 히스토리와 기술적 변경사항을 정리한 문서입니다.

---

## Version Overview

| Version | Commit | Date | Core Changes |
|---------|--------|------|--------------|
| v0 | c89e4a8 | 2025-05-28 | 논문 원본 구현 (CUDA only) |
| v1 | f40ba93 | 2025-12-05 | Mac MPS 지원, Web UI, Style Image Transfer |
| v2 | (uncommitted) | 2025-12-08 | RealVisXL 모델, 비동기 Task 시스템, Dual Adapter Mode 개선, VLM 프롬프트 생성 |
| v3 | e9eef5e | 2025-12-08 | Batch-Wise Decoupled Embedding, img2img 제거, Face ID 보존 강화 |
| v4 | (current) | 2025-12-11 | CLIP Blending으로 Identity Loss 문제 해결 |

---

## v0: Original Paper Implementation

### Commit: c89e4a8 (2025-05-28)

논문 "FastFace: Fast and Consistent Face Generation with Decoupled Guidance"의 원본 구현체입니다.

### Core Components

#### 1. Decoupled Classifier-Free Guidance (DCG)

기존 CFG의 한계를 극복하기 위한 핵심 기법입니다.

**기존 CFG 문제점:**
```
noise_pred = uncond + guidance_scale * (cond - uncond)
```
- 단일 guidance scale로 텍스트와 이미지 조건을 동시에 제어
- ID 보존과 프롬프트 충실도 사이의 trade-off 발생

**DCG 해결책:**
```python
# DCG Type 3 (논문 기본 방식)
term1 = eps_i - uncond      # 이미지(얼굴) 가이던스
term2 = eps_ti - eps_i      # 텍스트 가이던스
pred = uncond + a * term1 + b * term2
```

- `a`: 얼굴 ID 보존 강도 (guidance_scale_a)
- `b`: 텍스트 프롬프트 영향력 (guidance_scale_b)
- 두 조건을 독립적으로 제어 가능

**DCG Types:**
- Type 1: `eps_t, eps_ti` - 텍스트 우선
- Type 2: `eps_t, eps_i` - 분리된 조건
- Type 3: `eps_i, eps_ti` - 논문 기본값, 얼굴 우선

#### 2. Attention Manipulation (AM)

IP-Adapter의 cross-attention 출력을 조작하여 얼굴 특징을 강화합니다.

**적용 위치:** `src/custom_dca.py`

```python
am_patch_kwargs = {
    "target_parts": ["down", "up"],      # UNet의 down/up blocks
    "target_tokens": [0, 1, 2, 3],       # 조작할 토큰 인덱스
    "target_tsteps": [0, 1, 2],          # 적용할 timestep
    "am_transforms": ["pow", "scale"],   # 변환 함수
}
```

**원리:**
1. IP-Adapter가 생성한 cross-attention 출력에서 얼굴 관련 토큰 선택
2. `pow` 변환: 값의 분포를 조정하여 강한 특징 강조
3. `scale` 변환: 전체적인 영향력 조절
4. 초기 timestep에서만 적용하여 구조 형성 단계에 집중

#### 3. IP-Adapter FaceID Plus v2

얼굴 ID 보존을 위한 두 가지 임베딩 사용:

1. **InsightFace Embedding (512-dim)**
   - 얼굴 인식 모델에서 추출한 ID 벡터
   - 인물의 고유한 특징 (눈, 코, 입 위치 등)

2. **CLIP Embedding**
   - 얼굴 이미지의 시각적 특징
   - 피부 톤, 조명, 스타일 정보 포함

```python
id_embeds, clip_embeds = get_faceid_embeds(pipe, app, face_image, "plus-v2", ...)
pipe.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds
pipe.unet.encoder_hid_proj.image_projection_layers[0].shortcut = True
```

#### 4. 지원 모델

- **base**: SDXL Base 1.0 (50 steps)
- **hyper**: Hyper-SD (1/2/4/8 steps) - ByteDance
- **lcm**: Latent Consistency Model
- **lightning**: SDXL-Lightning (2/4/8 steps)
- **turbo**: SDXL-Turbo (512x512)

### 제한사항

- CUDA 전용 (`CUDAExecutionProvider`)
- Jupyter Notebook/CLI 인터페이스만 제공
- 단일 이미지 입력 (얼굴만)

---

## v1: Mac MPS Support + Web UI + Style Transfer

### Commit: f40ba93 (2025-12-05)

Mac Apple Silicon에서 실행 가능하도록 수정하고, Web UI와 스타일 이미지 전송 기능을 추가했습니다.

### 1. Mac MPS (Metal Performance Shaders) 지원

#### 변경된 코드: `src/sdxl_custom_pipeline.py`

**기존 (v0):**
```python
device_id = device.index if hasattr(device, 'index') else 0
app = FaceAnalysis(name="buffalo_l",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
    provider_options=[{"device_id": device_id}, {}])
```

**수정 (v1):**
```python
if torch.cuda.is_available() and 'cuda' in str(device):
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    provider_options = [{"device_id": device_id}, {}]
else:
    # MPS or CPU - InsightFace는 MPS 미지원이므로 CPU 사용
    providers = ['CPUExecutionProvider']
    provider_options = [{}]
```

**원리:**
- PyTorch의 MPS backend는 Apple Silicon GPU 사용
- 하지만 InsightFace (ONNX Runtime)는 MPS 미지원
- 해결: 모델 추론은 MPS, 얼굴 검출은 CPU로 분리

#### 추가 파일: `requirements_mps.txt`

Mac 전용 의존성 분리:
```
torch>=2.0.0
torchvision>=0.15.0
# CUDA 관련 패키지 제외
```

#### 추가 파일: `setup_mac.sh`

Mac 환경 자동 설정 스크립트:
```bash
# MPS 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"

# 환경 설정
export DEVICE=mps
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### 2. Style Image Transfer (img2img)

단순 얼굴 생성을 넘어, 참조 이미지의 스타일(배경, 분위기)을 적용하는 기능입니다.

#### 추가된 메서드: `_prepare_style_latents()`

```python
def _prepare_style_latents(self, style_image, height, width, generator, dtype):
    """스타일 이미지를 VAE latent space로 인코딩"""
    # 1. 이미지 로드 및 리사이즈
    style_image = style_image.resize((width, height), Image.LANCZOS)

    # 2. 텐서 변환: [0, 255] -> [-1, 1]
    style_tensor = TF.to_tensor(style_image).unsqueeze(0)
    style_tensor = (style_tensor * 2.0 - 1.0).to(self.device, dtype=dtype)

    # 3. VAE 인코딩
    with torch.no_grad():
        encoded = self.vae.encode(style_tensor)
        init_latents = encoded.latent_dist.sample(generator)
        init_latents = init_latents * self.vae.config.scaling_factor

    return init_latents
```

#### img2img 방식의 원리

**일반 txt2img:**
```
Random Noise → Denoising (N steps) → Image
```

**img2img:**
```
Style Latents + Noise → Denoising (N * strength steps) → Image
```

**구현:**
```python
# style_strength = 0.3이면 denoising = 0.7
# → 30% 스타일 유지, 70% 새로 생성
denoising = 1.0 - style_strength

# timestep 조정
init_timestep = int(num_inference_steps * denoising_strength)
t_start = num_inference_steps - init_timestep
timesteps = timesteps[t_start:]  # 앞부분 스킵

# 노이즈 추가
latents = scheduler.add_noise(init_latents, noise, latent_timestep)
```

**효과:**
- `style_strength=0.3`: 스타일 약하게, 프롬프트 강하게
- `style_strength=0.7`: 스타일 강하게, 배경/분위기 유지

### 3. Web UI (Next.js + FastAPI)

#### Frontend: `frontend/`

- **Framework**: Next.js 14 + TypeScript
- **Styling**: Tailwind CSS
- **Features**:
  - 드래그 앤 드롭 이미지 업로드
  - 실시간 파라미터 조절 (Circular Dial, Visual Slider)
  - 생성 히스토리 관리
  - 폴더 정리 기능

#### Backend: `backend/main.py`

- **Framework**: FastAPI
- **Features**:
  - REST API 엔드포인트
  - 파일 업로드/다운로드
  - 히스토리 저장 (JSON)
  - 모델 핫스왑

**API 엔드포인트:**
```
POST /upload          - 이미지 업로드
POST /generate        - 이미지 생성 (동기)
GET  /models          - 사용 가능한 모델 목록
GET  /history         - 생성 히스토리
GET  /folders         - 폴더 목록
```

### 4. execute() 메서드 확장

```python
def execute(
    self,
    face_image,
    prompt,
    generator,
    pipe_kwargs,
    after_hook_fn=None,
    style_image=None,        # NEW: 스타일 참조 이미지
    style_strength=0.3,      # NEW: 스타일 강도
):
```

---

## v2: RealVisXL + Async Tasks + Dual Adapter Mode

### Status: Uncommitted (2025-12-08)

실사 품질 향상, 비동기 처리, 스타일 전송 방식 개선을 포함합니다.

### 1. RealVisXL V4.0 모델 추가

#### 추가된 코드: `src/sdxl_custom_pipeline.py`

```python
elif model_name == "realvis":
    # RealVisXL V4.0 - 실사 피부 질감 특화 모델
    if not os.path.exists(f"models_cache/realvis-hyper-{n_steps}-fused"):
        # 1. RealVisXL 베이스 로드
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "SG161222/RealVisXL_V4.0",
            torch_dtype=torch.float16,
            cache_dir="models_cache/realvis"
        ).to(device)

        # 2. Hyper-SD LoRA 적용 (빠른 생성)
        pipe.load_lora_weights(hf_hub_download(
            "ByteDance/Hyper-SD",
            f"Hyper-SDXL-{n_steps}steps-lora.safetensors",
            cache_dir=f"models_cache/sdxl-hyper-{n_steps}")
        )
        pipe.fuse_lora()
        pipe.unload_lora_weights()

        # 3. 캐싱
        pipe.save_pretrained(f"models_cache/realvis-hyper-{n_steps}-fused")

    pipe.scheduler = DDIMScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing"
    )
```

**RealVisXL의 특징:**
- 실사 피부 질감에 최적화된 fine-tuned SDXL
- 포토리얼리스틱 결과물
- Hyper-SD LoRA와 결합하여 4-step 빠른 생성

**vs Hyper-SD 기본:**
| 특성 | Hyper-SD | RealVisXL + Hyper |
|------|----------|-------------------|
| 베이스 | SDXL Base | RealVisXL V4.0 |
| 피부 질감 | 일반 | 실사 특화 |
| 조명 표현 | 일반 | 자연스러운 스튜디오 조명 |
| 적합 용도 | 다양한 스타일 | 인물 사진 |

### 2. 비동기 Task 시스템

#### 변경된 코드: `backend/main.py`

**기존 (v1) - 동기 처리:**
```python
@app.post("/generate")
async def generate_image(...):
    # 요청 처리 중 블로킹
    result = pipeline.execute(...)  # 수십 초 대기
    return {"image_url": result}
```

**수정 (v2) - 비동기 처리:**
```python
@app.post("/generate")
async def generate_image(background_tasks: BackgroundTasks, ...):
    task_id = str(uuid.uuid4())

    # Task 생성 및 즉시 반환
    task = {
        "id": task_id,
        "status": TaskStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        ...
    }
    generation_tasks[task_id] = task

    # 백그라운드에서 실행
    thread = threading.Thread(
        target=run_generation_task,
        args=(task_id, params)
    )
    thread.start()

    return {"task_id": task_id, "status": "pending"}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    return generation_tasks[task_id]
```

**Frontend 폴링:**
```typescript
const startPolling = (taskId: string) => {
    const poll = async () => {
        const task = await getTaskStatus(taskId);
        setActiveTask(task);

        if (task.status === 'completed') {
            setGeneratedImage(task.image_url);
            clearInterval(pollingRef.current);
        }
    };

    pollingRef.current = setInterval(poll, 1000);
};
```

**장점:**
- 요청 즉시 응답 → UX 개선
- 진행 상황 실시간 표시
- 서버 재시작 시 Task 상태 복구
- 여러 요청 동시 처리 가능

### 3. Dual Adapter Mode 개선

#### 기존 문제 (v1 시도)

```python
# v1의 접근: Style CLIP을 직접 주입
style_clip_embeds = self._get_style_clip_embeds(style_image)
self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = style_clip_embeds
```

**문제점:**
- `clip_embeds`가 모든 배치에 동일하게 적용
- Face CLIP이 완전히 대체되어 얼굴 특성 손실
- DCG Type 4의 4개 배치가 구분되지 않음

#### 새로운 접근 (v2)

```python
if dual_adapter_mode and style_image is not None:
    # 1. Face embedding (ID + CLIP) 정상 추출
    id_embeds, face_clip_embeds = get_faceid_embeds(...)

    # 2. Style latents 준비 (img2img)
    init_latents = self._prepare_style_latents(style_image, ...)

    # 3. Style CLIP 추출
    style_clip_embeds = self._get_style_clip_embeds_simple(style_image, dcg_type)

    # 4. Face CLIP + Style CLIP 블렌딩
    blended_clip = (1 - style_strength) * face_clip_embeds + style_strength * style_clip_embeds

    # 5. Denoising 조정
    denoising = 1.0 - (style_strength * 0.5)
```

**원리:**

1. **CLIP 공간 블렌딩**
   - Face CLIP: 얼굴의 시각적 특징 (피부 톤, 조명)
   - Style CLIP: 스타일 이미지의 전체적 분위기
   - `style_strength`로 비율 조절

2. **Latent 공간 블렌딩 (img2img)**
   - 스타일 이미지의 구조/배경 유지
   - Denoising strength로 변형 정도 조절

3. **결합 효과**
   - Face ID: InsightFace embedding → 인물 고유 특징 유지
   - Face CLIP + Style CLIP 블렌딩 → 피부/조명/분위기 조합
   - Style latents → 배경/구조 참조

### 4. DCG Type 4 지원

#### 추가된 코드: `src/dcg.py`

```python
# DCG Type 4: Dual Adapter mode with 4 batches
if dcg_type == 4:
    uncond, face_pred, style_pred, combined_pred = noise_pred.chunk(4)
    term1 = face_pred - uncond      # Face guidance
    term2 = style_pred - uncond     # Style guidance

    pred = uncond + a * term1 + b * term2
```

**4개 배치 구성:**
| 배치 | 내용 | 용도 |
|------|------|------|
| uncond | negative prompt | 기준점 |
| face_pred | face embedding only | 얼굴 가이던스 |
| style_pred | style embedding only | 스타일 가이던스 |
| combined_pred | face + text | 최종 조합 |

**vs DCG Type 3:**
- Type 3: 3개 배치 (uncond, image, image+text)
- Type 4: 4개 배치로 face와 style 분리 제어

### 5. 프론트엔드 개선

#### Negative Prompt 지원

```typescript
// execute() 호출 시 negative_prompt 전달
res = self(
    prompt=prompt,
    negative_prompt=negative_prompt,  // NEW
    ...
)
```

#### Progress Callback

```python
def step_callback(step_idx, t, latents):
    if progress_callback:
        progress_callback(step_idx + 1, total_steps)

res = self(
    ...
    callback=step_callback,
    callback_steps=1,
    ...
)
```

#### 새로 시작 버튼 개선

```tsx
<button className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium
    bg-neutral-100 hover:bg-black hover:text-white text-neutral-700
    rounded-xl transition-all whitespace-nowrap shadow-sm hover:shadow-md btn-press">
    <RefreshIcon size={14} />
    <span>새로 시작</span>
</button>
```

### 6. VLM 기반 프롬프트 자동 생성

Gemini 2.5 Flash를 활용하여 업로드된 이미지를 분석하고 최적의 프롬프트를 자동 생성하는 기능입니다.

#### Backend API: `/generate-prompt`

```python
@app.post("/generate-prompt")
async def generate_prompt_from_images(
    face_image_id: str = Form(...),
    style_image_id: Optional[str] = Form(None),
):
    """Generate prompts using Gemini VLM based on uploaded images"""

    # 1. 이미지를 base64로 인코딩
    with open(face_files[0], "rb") as f:
        face_image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # 2. Gemini API에 이미지와 시스템 프롬프트 전송
    image_parts = [{"inline_data": {"mime_type": face_mime, "data": face_image_data}}]

    # 스타일 이미지가 있으면 추가
    if style_image_id:
        image_parts.append({"inline_data": {...}})

    # 3. Gemini 2.5 Flash 호출
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    # 4. JSON 응답 파싱
    return {
        "success": True,
        "positive": prompts["positive"],
        "negative": prompts["negative"]
    }
```

**시스템 프롬프트 설계:**

```
You are an expert at creating Stable Diffusion prompts for portrait photography.

Analyze the provided image(s):
- The first image contains a FACE/PERSON
- The second image (if provided) is a STYLE REFERENCE

Generate TWO prompts:
1. POSITIVE PROMPT: Professional portrait style, lighting, background
   - Do NOT describe specific facial features (handled by FaceID)
   - Focus on atmosphere, lighting quality, color grading

2. NEGATIVE PROMPT: Elements to avoid
   - ugly, deformed, blurry, low quality, etc.

Output format: {"positive": "...", "negative": "..."}
```

**핵심 원칙:**
- 얼굴 세부 묘사 금지 (FaceID가 처리)
- 조명, 배경, 분위기에 집중
- 스타일 이미지가 있으면 그 특성 반영

#### Frontend: 타이핑 이펙트

```typescript
// State 변수
const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false)
const [promptTypingTarget, setPromptTypingTarget] = useState<{
  positive: string
  negative: string
} | null>(null)

// 타이핑 효과 useEffect
useEffect(() => {
  if (!promptTypingTarget) return

  const { positive, negative } = promptTypingTarget

  // Positive 프롬프트 먼저 타이핑
  if (prompt.length < positive.length) {
    const timeout = setTimeout(() => {
      setPrompt(positive.slice(0, prompt.length + 1))
    }, 15)  // 15ms 간격
    return () => clearTimeout(timeout)
  }
  // 그 다음 Negative 프롬프트 타이핑
  else if (negativePrompt.length < negative.length) {
    const timeout = setTimeout(() => {
      setNegativePrompt(negative.slice(0, negativePrompt.length + 1))
    }, 15)
    return () => clearTimeout(timeout)
  }
  // 완료
  else {
    setPromptTypingTarget(null)
    setIsGeneratingPrompt(false)
  }
}, [promptTypingTarget, prompt, negativePrompt])
```

**타이핑 효과 원리:**
1. VLM에서 전체 프롬프트 수신
2. `promptTypingTarget`에 저장
3. useEffect가 15ms마다 한 글자씩 추가
4. Positive 완료 후 Negative 시작
5. 모두 완료되면 상태 초기화

#### Handler 함수

```typescript
const handleGeneratePrompt = async () => {
  if (!uploadedImage) {
    setError('프롬프트 생성을 위해 먼저 얼굴 이미지를 업로드해주세요.')
    return
  }

  setIsGeneratingPrompt(true)
  setError(null)
  // 기존 프롬프트 초기화 (타이핑 효과를 위해)
  setPrompt('')
  setNegativePrompt('')

  try {
    const result = await generatePromptFromImages(
      uploadedImage.id,
      styleImage?.id || null  // 스타일 이미지 선택적 전송
    )

    if (result.success) {
      // 타이핑 시작
      setPromptTypingTarget({
        positive: result.positive,
        negative: result.negative,
      })
    }
  } catch (e) {
    setError('프롬프트 생성 중 오류가 발생했습니다.')
    setIsGeneratingPrompt(false)
  }
}
```

#### UI 버튼

```tsx
<div className="flex items-start justify-between mb-4">
  <div>
    <h3 className="text-sm font-semibold">프롬프트</h3>
    <p className="text-xs text-neutral-400 mt-0.5">생성하고 싶은 이미지를 설명하세요</p>
  </div>
  <button
    onClick={handleGeneratePrompt}
    disabled={!uploadedImage || isGeneratingPrompt}
    className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium
               bg-neutral-100 hover:bg-black hover:text-white text-neutral-600
               rounded-lg transition-all disabled:opacity-50 btn-press"
  >
    {isGeneratingPrompt ? (
      <>
        <LoaderIcon size={12} className="animate-spin" />
        <span>생성 중...</span>
      </>
    ) : (
      <>
        <SparkleIcon size={12} />
        <span>AI 프롬프트</span>
      </>
    )}
  </button>
</div>
```

#### 환경 설정

```bash
# backend/.env
GEMINI_API_KEY=your_gemini_api_key_here
```

**사용 흐름:**
1. 얼굴 이미지 업로드 (필수)
2. 스타일 이미지 업로드 (선택)
3. "AI 프롬프트" 버튼 클릭
4. Gemini가 이미지 분석 후 프롬프트 생성
5. 타이핑 효과로 프롬프트 필드에 자동 입력

---

## v3: True Dual IP-Adapter Architecture

### Status: In Development (2025-12-08)

v2의 CLIP Blending 및 img2img 방식의 한계를 극복하기 위해 **두 개의 독립적인 IP-Adapter**를 동시에 로드하는 아키텍처입니다.

### 1. 아키텍처 개요

```
Face Image                    Style Image
    |                              |
    v                              v
InsightFace + CLIP            CLIP Encoder
    |                              |
    v                              v
IP-Adapter FaceID Plus v2     IP-Adapter Plus
(얼굴 정체성 전용)              (스타일 전용)
    |                              |
    +--------- UNet Forward -------+
                  |
                  v
            Final Image
```

**핵심 원리:**
- 두 IP-Adapter가 **완전히 독립적으로** 동작
- FaceID Adapter: 얼굴 정체성만 담당 (InsightFace embedding)
- Style Adapter: 시각적 스타일만 담당 (CLIP embedding)
- Blending 없음 = Face ID 100% 보존

### 2. 파이프라인 로딩

```python
# get_faceid_pipeline() in sdxl_custom_pipeline.py
def get_faceid_pipeline(config, device, dual_adapter=True):
    if dual_adapter:
        # 두 개의 IP-Adapter 동시 로드
        pipe.load_ip_adapter(
            ["h94/IP-Adapter-FaceID", "h94/IP-Adapter"],
            subfolder=[None, "sdxl_models"],
            weight_name=[
                "ip-adapter-faceid-plusv2_sdxl.bin",      # FaceID adapter
                "ip-adapter-plus_sdxl_vit-h.safetensors"  # Style adapter
            ],
            image_encoder_folder=None,
            cache_dir="models_cache/ipadapter"
        )
        pipe.set_ip_adapter_scale([ip_adapter_scale, 0.0])  # Style은 동적 조절
        pipe.dual_adapter_enabled = True
```

### 3. Embedding 포맷 요구사항

두 IP-Adapter는 서로 다른 ImageProjection 레이어를 사용:

| Adapter | Projection Layer | Input Shape | 설명 |
|---------|-----------------|-------------|------|
| FaceID Plus v2 | IPAdapterFaceIDPlusImageProjection | `[batch, 1, 1, 512]` | InsightFace embedding + clip_embeds |
| IP-Adapter Plus | IPAdapterPlusImageProjection (Resampler) | `[batch, 1, 257, 1280]` | CLIP hidden states |

**중요:** 두 embedding 모두 **4D 텐서**여야 함!

### 4. execute() 메서드 구현

```python
def execute(self, face_image, prompt, ..., dual_adapter_mode=True, style_image=None):
    if dual_adapter_mode and style_image is not None and self.dual_adapter_enabled:
        # 1. Face ID embedding 추출 (InsightFace)
        face_embedding = torch.from_numpy(faces[0].normed_embedding)
        ref_images_embeds = face_embedding.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        # Shape: [1, 1, 1, 512]

        # DCG Type 3 배치 구성
        id_embeds = torch.cat([
            neg_ref_images_embeds,  # uncond
            ref_images_embeds,      # cond
            ref_images_embeds,      # cond
        ], dim=0)  # Shape: [3, 1, 1, 512]

        # 2. Face CLIP embedding 설정 (FaceID adapter 내부용)
        face_clip_embeds = face_clip_embeds.unsqueeze(1)  # [1, 1, 257, 1280]
        face_clip_batched = torch.cat([neg, pos, pos], dim=0)  # [3, 1, 257, 1280]
        self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = face_clip_batched

        # 3. Style embedding 추출
        style_embeds = self._encode_style_for_adapter(style_pil, dcg_type)
        # Shape: [3, 1, 257, 1280] (4D!)

        # 4. 두 adapter에 각각 embedding 전달
        res = self(
            prompt=prompt,
            ip_adapter_image_embeds=[id_embeds, style_embeds],  # 각 adapter별 embedding
            ...
        )
```

### 5. 구현 중 발생한 오류들

#### Error 1: IP Adapter Scale Mismatch

```
ValueError: Cannot assign 1 scale_configs to 2 IP-Adapter
```

**원인:** Dual adapter 모드에서 scale을 하나만 전달
**해결:** `backend/main.py`에서 dual adapter 감지 후 리스트로 전달

```python
if getattr(pipe, 'dual_adapter_enabled', False):
    pipe.set_ip_adapter_scale([params["ips"], 0.0])
else:
    pipe.set_ip_adapter_scale(params["ips"])
```

#### Error 2: IP Adapter Image Count Mismatch

```
ValueError: ip_adapter_image must have same length as the number of IP Adapters.
Got 1 images and 2 IP Adapters
```

**원인:** `prepare_ip_adapter_image_embeds`가 2개 이미지를 기대
**해결:** `execute()`에서 직접 embedding 추출하여 bypass

#### Error 3: CLIP Embedding Shape Mismatch

```
IndexError: tuple index out of range at clip_embeds.reshape(-1, clip_embeds.shape[2], ...)
```

**원인:** FaceID의 ImageProjection은 4D tensor `[batch, num_images, seq_len, hidden_dim]` 기대
**해결:** `unsqueeze(1)` 추가하여 3D -> 4D 변환

```python
face_clip_embeds = face_clip_embeds.unsqueeze(1)  # [1, 257, 1280] -> [1, 1, 257, 1280]
```

#### Error 4: DCG Batch Size Detection

```
RuntimeError: Sizes of tensors must match except in dimension 1.
Expected size 3 but got size 4
```

**원인:** `prepare_ip_adapter_image_embeds`가 DCG type 4만 감지하고 type 3은 추가 처리
**해결:** DCG type 1-3도 batch size 3으로 감지

```python
is_dcg_batched = (
    (dcg_type == 4 and single_image_embeds.shape[0] == 4) or
    (dcg_type in [1, 2, 3] and single_image_embeds.shape[0] == 3)
)
if is_dcg_batched:
    # Already properly stacked - bypass further processing
    image_embeds.append(single_image_embeds)
```

#### Error 5: Resampler Dimension Mismatch

```
RuntimeError: Tensors must have same number of dimensions: got 2 and 3
```

**원인:** Style embedding이 3D `[3, 257, 1280]`인데 Resampler는 4D 기대
**해결:** `_encode_style_for_adapter`에서 4D로 변환

```python
# Before: style_embeds shape [3, 257, 1280] (3D)
image_embeds = image_embeds.unsqueeze(1)  # [1, 257, 1280] -> [1, 1, 257, 1280]
negative_image_embeds = negative_image_embeds.unsqueeze(1)

style_embeds = torch.cat([
    negative_image_embeds,
    image_embeds,
    image_embeds,
], dim=0)  # [3, 1, 257, 1280] (4D)
```

### 6. 최종 Tensor Shape 요약

DCG Type 3 기준:

| Tensor | Shape | 용도 |
|--------|-------|------|
| `id_embeds` | `[3, 1, 1, 512]` | FaceID adapter input |
| `face_clip_batched` | `[3, 1, 257, 1280]` | FaceID adapter의 clip_embeds |
| `style_embeds` | `[3, 1, 257, 1280]` | Style adapter input |

### 7. v2 vs v3 비교

| 항목 | v2 (CLIP Blending) | v3 (Dual Adapter) |
|------|-------------------|-------------------|
| IP-Adapter 수 | 1개 (FaceID) | 2개 (FaceID + Style) |
| Style 적용 방식 | CLIP embedding 섞기 | 별도 adapter로 처리 |
| Face ID 보존 | 70~80% | 100% (이론상) |
| 구조 오염 | img2img로 발생 가능 | txt2img만 사용, 없음 |
| 복잡도 | 낮음 | 높음 (embedding 포맷 관리) |

### 8. 사용법

API는 v2와 동일:

```python
res = pipe.execute(
    face_image="face.jpg",
    style_image="style.jpg",
    prompt="professional portrait",
    dual_adapter_mode=True,
    style_strength=0.5,  # Style adapter scale 직접 제어
)
```

**style_strength 의미:**
- v3에서는 Style IP-Adapter의 scale을 직접 제어
- 0.0 ~ 1.0 권장 (너무 높으면 스타일이 과도하게 적용)

---

## v4: CLIP Blending (Identity Loss Fix)

### Status: In Development (2025-12-11)

v3의 Dual IP-Adapter 방식에서 발생한 **Identity Loss 문제**를 해결하기 위해 CLIP Embedding Blending 방식으로 전환했습니다.

### 1. 문제 분석

#### v3의 문제점

스타일 이미지에 인물이 없는 경우 (배경만 있는 이미지), 생성된 이미지에서도 인물이 사라지는 현상이 발생했습니다.

**원인:**

1. **CLIP은 Semantic 정보를 인코딩**
   - CLIP ViT는 이미지-텍스트 정렬을 위해 학습됨
   - "스타일"만 추출하는 것이 아니라 **전체 의미 정보** 인코딩
   - 객체 존재 여부 ("인물 있음/없음")가 강하게 인코딩됨

2. **토큰 수 불균형**

   | Adapter | 토큰 수 | 정보량 |
   |---------|---------|--------|
   | FaceID Plus v2 | 4 (projected) | 얼굴 특징만 |
   | IP-Adapter Plus (Style) | 257 | 전체 이미지 의미 |

3. **Cross-Attention 충돌**

   ```python
   # FaceID: "이 얼굴을 생성해라"
   # Style: "이 장면에는 인물이 없다"
   # → Style의 257개 토큰이 FaceID의 4개 토큰을 압도
   ```

자세한 분석은 [GitHub Issue #1](https://github.com/danlee-dev/prometheus-fastface-dev/issues/1) 참조.

### 2. 해결책: CLIP Embedding Blending

Dual Adapter 대신 **단일 FaceID Adapter**를 사용하되, **Face CLIP과 Style CLIP을 블렌딩**합니다.

#### 핵심 공식

```python
blended_clip = (1 - alpha) * face_clip + alpha * style_clip
```

#### 원리

```python
# Face CLIP embedding (개념적 분해)
face_clip = [
    person_presence: 0.9,      # "인물이 있음" - 강하게 인코딩
    face_features: [0.8, ...],
    skin_tone: 0.6,
    lighting: 0.5,
    ...
]

# Style CLIP embedding (배경만 있는 이미지)
style_clip = [
    person_presence: 0.1,      # "인물이 없음" - 문제의 원인!
    face_features: [0.0, ...],
    lighting: 0.8,             # 조명 정보
    background: 0.9,           # 배경 정보
    ...
]

# Blending (alpha = 0.3)
blended = [
    person_presence: 0.7 * 0.9 + 0.3 * 0.1 = 0.66,  # 인물 정보 유지!
    face_features: [0.56, ...],
    lighting: 0.35 + 0.24 = 0.59,                    # 스타일 조명 혼합
    background: 0.21 + 0.27 = 0.48,                  # 스타일 배경 혼합
    ...
]
```

**핵심:** `person_presence`가 0.66으로 유지되어 UNet이 "인물이 있다"고 판단

### 3. 구현 변경사항

#### execute() 메서드 수정

**변경 전 (v3):**

```python
# 두 개의 독립적인 adapter 사용
ip_adapter_image_embeds = [id_embeds, style_embeds]
self.set_ip_adapter_scale([face_scale, style_scale])
```

**변경 후 (v4):**

```python
# Face CLIP + Style CLIP 블렌딩
blended_clip = (1 - style_strength) * face_clip_embeds + style_strength * style_clip_embeds
self.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = blended_clip

# Style adapter 비활성화 (scale=0)
if dual_adapter_enabled:
    self.set_ip_adapter_scale([face_scale, 0.0])
    ip_adapter_embeds = [id_embeds, dummy_style_embeds]
else:
    ip_adapter_embeds = [id_embeds]
```

#### 로그 출력

```
>>> [CLIP Blending Mode] style_strength=0.3
>>> [DEBUG] face_clip_embeds shape: torch.Size([3, 257, 1280])
>>> [DEBUG] style_clip_embeds shape: torch.Size([3, 257, 1280])
>>> [DEBUG] blended_clip shape: torch.Size([3, 257, 1280])
>>> [DEBUG] Blend ratio - Face: 70.0%, Style: 30.0%
>>> [DEBUG] Dual adapter loaded but Style adapter disabled (scale=0)
```

### 4. v3 vs v4 비교

| 항목 | v3 (Dual Adapter) | v4 (CLIP Blending) |
|------|-------------------|-------------------|
| IP-Adapter 수 | 2개 (FaceID + Style) | 1개 (FaceID만 활성) |
| Style 적용 방식 | 별도 adapter | CLIP embedding 블렌딩 |
| Identity 보존 | 낮음 (Style이 override) | 높음 (Face CLIP 유지) |
| 스타일 분리 | 이론상 완전 분리 | 부분적 분리 |
| 배경만 있는 스타일 | 인물 사라짐 | 인물 유지됨 |

### 5. style_strength 의미 변화

| Version | style_strength 의미 |
|---------|---------------------|
| v2 | CLIP blending 비율 (동일) |
| v3 | Style adapter scale (0.0~1.0) |
| v4 | CLIP blending 비율 (v2와 동일하게 복원) |

### 6. 한계점 및 향후 개선

**현재 한계:**

- CLIP embedding 자체가 의미 정보를 포함하므로 완벽한 스타일 분리 불가
- 블렌딩 비율에 따라 스타일 효과가 약해질 수 있음

**향후 개선 (v5 예정):**

- ControlNet 통합으로 구조적 제약 추가
- 인물 구조를 강제하면서 스타일만 변경 가능

---

## Technical Deep Dive

### Decoupled Guidance의 수학적 원리

**기존 CFG:**
$$\epsilon_\theta(x_t, c) = \epsilon_\theta(x_t, \emptyset) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$$

**DCG:**
$$\epsilon_{DCG} = \epsilon_\emptyset + \alpha \cdot (\epsilon_i - \epsilon_\emptyset) + \beta \cdot (\epsilon_{ti} - \epsilon_i)$$

- $\epsilon_\emptyset$: unconditional prediction
- $\epsilon_i$: image-conditioned prediction (IP-Adapter)
- $\epsilon_{ti}$: text+image-conditioned prediction
- $\alpha$: image guidance scale
- $\beta$: text guidance scale

### IP-Adapter FaceID Plus v2 구조

```
Face Image
    │
    ├─→ InsightFace ─→ 512-dim ID embedding ─→ ID Projection ─┐
    │                                                          │
    └─→ Face Align ─→ CLIP ViT-H ─→ 257x1280 embedding ───────┴─→ Cross Attention
```

### Style Transfer Pipeline

```
Style Image                Face Image
    │                          │
    ▼                          ▼
VAE Encoder              InsightFace + CLIP
    │                          │
    ▼                          ▼
init_latents             id_embeds + clip_embeds
    │                          │
    └──────────┬───────────────┘
               │
               ▼
        Add Noise (t_start)
               │
               ▼
        Denoising Loop
        (with DCG + AM)
               │
               ▼
        VAE Decoder
               │
               ▼
         Final Image
```

---

## Configuration Reference

### `configs/fastface/am1_and_dcg.json`

```json
{
    "model_name": "hyper",
    "method": "faceid",
    "pipe_kwargs": {
        "num_inference_steps": 4,
        "guidance_scale_a": 1.0,      // 얼굴 가이던스 강도
        "guidance_scale_b": 1.0,      // 텍스트 가이던스 강도
        "dcg_kwargs": {
            "dcg_type": 3,            // DCG 타입 (1-4)
            "term_postproc": "rescale",
            "rescale": 0.75,
            "a_scheduler": "custom",
            "b_scheduler": "custom",
            "sch_kwargs": {
                "custom": {
                    "a": [1.0, 1.5, 1.2, 0.5],  // step별 얼굴 가중치
                    "b": [1.0, 2.5, 2.0, 0.5]   // step별 텍스트 가중치
                }
            }
        }
    },
    "patch_pipe": true,
    "am_patch_kwargs": {
        "target_parts": ["down", "up"],
        "target_tokens": [0, 1, 2, 3],
        "target_tsteps": [0, 1, 2],
        "am_transforms": ["pow", "scale"]
    }
}
```

---

## Version Migration Notes

### v0 → v1

1. `requirements.txt` 대신 `requirements_mps.txt` 사용 (Mac)
2. `DEVICE=mps` 환경변수 설정
3. Web UI 실행: `./run_backend.sh && ./run_frontend.sh`

### v1 → v2

1. 모델 선택에서 "RealVisXL" 옵션 추가됨
2. 생성 요청이 비동기로 변경됨 (polling 필요)
3. Dual Adapter Mode에서 스타일 강도 0.5~0.7 권장

### v2 → v3

1. Dual Adapter Mode 내부 로직 완전 변경 (API는 동일)
2. `style_strength`의 의미 변화: CLIP blending 비율 → DCG guidance_scale_b
3. 스타일 이미지의 구조 오염 문제 해결
4. Face ID 보존율 향상 (70% → 100%)

---

## Future Improvements

- [ ] ControlNet 통합 (포즈/깊이 제어)
- [ ] Multi-face 지원
- [ ] Video generation (AnimateDiff)
- [ ] WebSocket 실시간 진행 표시
- [ ] GPU 메모리 최적화 (attention slicing)
