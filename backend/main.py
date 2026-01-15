import os
import sys
import uuid
import json
import gc
import shutil
import asyncio
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

load_dotenv()

# Configuration from environment
PORT = int(os.getenv("PORT", 8007))
HOST = os.getenv("HOST", "0.0.0.0")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3007").split(",")
DEVICE = os.getenv("DEVICE", "auto")  # auto, cuda, mps, or cpu
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "hyper")
CONFIG_PATH = os.getenv("CONFIG_PATH", "configs/fastface/am1_and_dcg.json")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
DATA_DIR = os.getenv("DATA_DIR", "data")
HISTORY_FILE = os.path.join(DATA_DIR, "history.json")
FOLDERS_FILE = os.path.join(DATA_DIR, "folders.json")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Global pipeline instance
pipeline = None
current_model = None  # Format: "model_name" or "model_name_steps" for step-specific models

# Models that have step-specific LoRAs/UNets
STEP_SPECIFIC_MODELS = {"hyper", "realvis", "lightning"}
# Valid steps for each model type
VALID_STEPS = {
    "hyper": [1, 2, 4, 8, 12, 16, 20],
    "realvis": [1, 2, 4, 8, 12, 16, 20],
    "lightning": [2, 4, 8, 12, 16],
    "turbo": [1, 2, 4],  # turbo is fast, doesn't need many steps
    "lcm": [2, 4, 6, 8, 12],
    "base": [20, 30, 50],  # base needs more steps
}

# Task tracking for generation jobs
TASKS_FILE = os.path.join(DATA_DIR, "tasks.json")
generation_tasks: Dict[str, Dict[str, Any]] = {}
task_lock = threading.Lock()

# Create directories on import
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models_cache", exist_ok=True)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def load_tasks() -> Dict[str, Dict[str, Any]]:
    """Load tasks from file"""
    if os.path.exists(TASKS_FILE):
        try:
            with open(TASKS_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_tasks(tasks: Dict[str, Dict[str, Any]]):
    """Save tasks to file"""
    with open(TASKS_FILE, "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def update_task(task_id: str, updates: Dict[str, Any]):
    """Thread-safe task update"""
    global generation_tasks
    with task_lock:
        if task_id in generation_tasks:
            generation_tasks[task_id].update(updates)
            generation_tasks[task_id]["updated_at"] = datetime.now().isoformat()
            save_tasks(generation_tasks)


# History and Folder models
class HistorySettings(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_name: str
    prompt: str
    ips: float
    lora_scale: float
    seed: int


class HistoryItem(BaseModel):
    id: str
    title: str
    created_at: str
    folder_id: Optional[str] = None
    input_image_url: str
    output_image_url: str
    settings: HistorySettings


class Folder(BaseModel):
    id: str
    name: str
    created_at: str
    order: int = 0


def load_history() -> List[dict]:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []


def save_history(history: List[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def load_folders() -> List[dict]:
    if os.path.exists(FOLDERS_FILE):
        with open(FOLDERS_FILE, "r") as f:
            return json.load(f)
    return []


def save_folders(folders: List[dict]):
    with open(FOLDERS_FILE, "w") as f:
        json.dump(folders, f, indent=2, ensure_ascii=False)


class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    prompt: str
    model_name: str = "hyper"
    ips: float = 0.8
    lora_scale: float = 0.6
    seed: int = 42


class InferenceResponse(BaseModel):
    success: bool
    image_url: Optional[str] = None
    error: Optional[str] = None


def get_device():
    """Get the compute device based on DEVICE env var or auto-detection."""
    device_setting = DEVICE.lower()

    # Explicit device setting
    if device_setting == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        print(">>> [WARNING] CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    elif device_setting == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print(">>> [WARNING] MPS requested but not available, falling back to CPU")
        return torch.device("cpu")
    elif device_setting == "cpu":
        return torch.device("cpu")

    # Auto-detection (default)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_cache_key(model_name: str, inference_steps: int) -> str:
    """Generate cache key based on model and steps (for step-specific models)"""
    if model_name in STEP_SPECIFIC_MODELS:
        return f"{model_name}_{inference_steps}"
    return model_name


def get_valid_steps(model_name: str, requested_steps: int) -> int:
    """Return the closest valid step count for the given model"""
    valid = VALID_STEPS.get(model_name, [4])
    if requested_steps in valid:
        return requested_steps
    # Find closest valid step
    return min(valid, key=lambda x: abs(x - requested_steps))


def load_pipeline(model_name: str, inference_steps: int = 4, use_tiny_vae: bool = False, progress_callback=None):
    global pipeline, current_model

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # Validate and adjust steps for the model type
    valid_steps = get_valid_steps(model_name, inference_steps)
    if valid_steps != inference_steps:
        log(f"Adjusted steps from {inference_steps} to {valid_steps} for {model_name}")
        inference_steps = valid_steps

    # Generate cache key (model_name for non-step-specific, model_name_steps for step-specific)
    # Include VAE type in cache key to properly cache different configurations
    vae_suffix = "_tiny" if use_tiny_vae else "_full"
    cache_key = get_cache_key(model_name, inference_steps) + vae_suffix

    if pipeline is not None and current_model == cache_key:
        log(f"Using cached {model_name} model (steps={inference_steps}, VAE={'TinyVAE' if use_tiny_vae else 'Full'})...")
        return pipeline

    # Clean up existing pipeline
    if pipeline is not None:
        log("Cleaning up previous model...")
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    import src.sdxl_custom_pipeline as sdxl_pipeline
    from diffusers import AutoencoderTiny

    device = get_device()
    config_path = Path(__file__).parent.parent / CONFIG_PATH

    log("Loading configuration...")
    with open(config_path, "r") as f:
        conf = json.load(f)

    conf["model_name"] = model_name
    conf["pipe_kwargs"]["num_inference_steps"] = inference_steps  # Use user-selected steps

    default_hw = (1024, 1024)
    if conf["reset_gs_to_default"]:
        if model_name == "lcm":
            conf["pipe_kwargs"]["guidance_scale_a"] = 1.5
            conf["pipe_kwargs"]["guidance_scale_b"] = 1.5
        elif model_name == "base":
            conf["pipe_kwargs"]["guidance_scale_a"] = 5.
            conf["pipe_kwargs"]["guidance_scale_b"] = 5.
        else:
            conf["pipe_kwargs"]["guidance_scale_a"] = 1.
            conf["pipe_kwargs"]["guidance_scale_b"] = 1.

    if model_name == "turbo":
        conf["pipe_kwargs"]["height"] = 512
        conf["pipe_kwargs"]["width"] = 512
    else:
        conf["pipe_kwargs"]["height"] = default_hw[0]
        conf["pipe_kwargs"]["width"] = default_hw[1]

    log(f"Initializing {model_name} pipeline (steps={inference_steps})...")
    log(f"Loading UNet and text encoders...")
    pipe = sdxl_pipeline.name2pipe["faceid"](conf, device)

    # VAE selection: TinyVAE (fast) vs Full VAE (quality)
    if use_tiny_vae:
        log("Loading TinyVAE for faster decoding...")
        tiny_vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        pipe.vae = tiny_vae
        log("TinyVAE loaded - faster decoding, slight quality trade-off")
    else:
        # Full VAE from base model provides better quality
        log("Using Full VAE decoder for maximum quality...")

    log("Optimizing memory...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    pipe._config = conf
    pipeline = pipe
    current_model = cache_key  # Cache with step info for step-specific models

    log("Model ready!")
    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global generation_tasks
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models_cache", exist_ok=True)

    # Load existing tasks and mark running ones as failed (server restart)
    generation_tasks = load_tasks()
    for task_id, task in generation_tasks.items():
        if task.get("status") in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            task["status"] = TaskStatus.FAILED
            task["error"] = "Server restarted during generation"
            task["updated_at"] = datetime.now().isoformat()
    save_tasks(generation_tasks)

    yield
    # Shutdown
    global pipeline
    if pipeline is not None:
        del pipeline


app = FastAPI(
    title="Prometheus API",
    description="FastFace Image Generation API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files for generated images
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")


@app.get("/")
async def root():
    return {"status": "ok", "message": "Prometheus API is running"}


@app.get("/health")
async def health_check():
    device = get_device()
    return {
        "status": "healthy",
        "device": str(device),
        "model_loaded": pipeline is not None,
        "current_model": current_model
    }


@app.get("/models")
async def get_models():
    return {
        "models": [
            {
                "id": "hyper",
                "name": "Hyper-SD",
                "description": "고품질 빠른 생성",
                "default": True,
                "valid_steps": VALID_STEPS["hyper"],
                "default_steps": 4,
            },
            {
                "id": "realvis",
                "name": "RealVisXL",
                "description": "실사 피부 질감 특화",
                "valid_steps": VALID_STEPS["realvis"],
                "default_steps": 4,
            },
            {
                "id": "lightning",
                "name": "SDXL-Lightning",
                "description": "Lightning 모델",
                "valid_steps": VALID_STEPS["lightning"],
                "default_steps": 4,
            },
            {
                "id": "lcm",
                "name": "LCM",
                "description": "Latent Consistency 모델",
                "valid_steps": VALID_STEPS["lcm"],
                "default_steps": 4,
            },
            {
                "id": "turbo",
                "name": "SDXL-Turbo",
                "description": "512x512 빠른 생성",
                "valid_steps": VALID_STEPS["turbo"],
                "default_steps": 2,
            },
        ]
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".jpg"
    filename = f"{file_id}{ext}"
    filepath = Path(UPLOAD_DIR) / filename

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "success": True,
        "file_id": file_id,
        "filename": filename,
        "url": f"/uploads/{filename}"
    }


def run_generation_task(task_id: str, params: Dict[str, Any]):
    """Background task for image generation"""
    from src.custom_dca import patch_pipe, reset_patched_unet

    def log_progress(message: str):
        """Update task with progress message"""
        update_task(task_id, {"progress_message": message})
        print(f">>> [TASK {task_id}] {message}")

    try:
        update_task(task_id, {"status": TaskStatus.RUNNING, "progress_message": "Starting generation..."})

        log_progress("Locating uploaded images...")
        upload_path = Path(UPLOAD_DIR)
        image_files = list(upload_path.glob(f"{params['image_id']}.*"))
        if not image_files:
            update_task(task_id, {
                "status": TaskStatus.FAILED,
                "error": "Uploaded image not found"
            })
            return

        image_path = str(image_files[0])

        # Find style image if provided
        style_image_path = None
        style_image_size = None  # v7.5: For matching output size to style image
        if params.get("style_image_id"):
            style_files = list(upload_path.glob(f"{params['style_image_id']}.*"))
            if style_files:
                style_image_path = str(style_files[0])
                log_progress("Style image found, preparing dual adapter mode...")
                # v7.5: Get style image dimensions for output size
                from PIL import Image as PILImage
                with PILImage.open(style_image_path) as style_img:
                    style_image_size = style_img.size  # (width, height)
                    log_progress(f"Style image size: {style_image_size[0]}x{style_image_size[1]}")

        # Load pipeline with progress updates (pass inference_steps for step-specific models)
        pipe = load_pipeline(
            params["model_name"],
            inference_steps=params["inference_steps"],
            use_tiny_vae=params.get("use_tiny_vae", False),
            progress_callback=log_progress
        )
        conf = pipe._config
        device = get_device()

        log_progress("Configuring generation parameters...")
        # Set parameters
        if conf["method"] == "faceid" and "faceid_lora_scale" in conf:
            if "faceid_0" in pipe.unet.peft_config:
                pipe.set_adapters(["faceid_0"], params["lora_scale"])

        # Dual adapter mode: set scales for both [FaceID, Style]
        # Style scale will be dynamically adjusted in execute() based on style_strength
        if getattr(pipe, 'dual_adapter_enabled', False):
            pipe.set_ip_adapter_scale([params["ips"], 0.0])  # Style scale set in execute()
        else:
            pipe.set_ip_adapter_scale(params["ips"])
        conf["pipe_kwargs"]["num_inference_steps"] = params["inference_steps"]

        # v7.5: Use style image dimensions for output size (if available)
        if style_image_size:
            # Constrain to reasonable limits and make divisible by 8
            max_size = 1536  # Maximum dimension
            min_size = 512   # Minimum dimension

            orig_w, orig_h = style_image_size

            # Scale down if too large, keeping aspect ratio
            if max(orig_w, orig_h) > max_size:
                scale = max_size / max(orig_w, orig_h)
                orig_w = int(orig_w * scale)
                orig_h = int(orig_h * scale)

            # Scale up if too small, keeping aspect ratio
            if min(orig_w, orig_h) < min_size:
                scale = min_size / min(orig_w, orig_h)
                orig_w = int(orig_w * scale)
                orig_h = int(orig_h * scale)

            # Make divisible by 8 (required for VAE)
            target_w = (orig_w // 8) * 8
            target_h = (orig_h // 8) * 8

            conf["pipe_kwargs"]["width"] = target_w
            conf["pipe_kwargs"]["height"] = target_h
            log_progress(f"Output size set to {target_w}x{target_h} (matching style image)")

        if conf["patch_pipe"]:
            log_progress("Applying attention manipulation patches...")
            patch_pipe(pipe, **conf["am_patch_kwargs"])

        generator = torch.Generator(device=device).manual_seed(params["seed"])

        log_progress("Extracting face embedding...")
        log_progress(f"[DEBUG] dual_adapter_mode={params.get('dual_adapter_mode')}, style_image_path={style_image_path}")

        with torch.no_grad():
            result = pipe.execute(
                image_path,
                params["prompt"],
                generator,
                conf["pipe_kwargs"],
                after_hook_fn=reset_patched_unet if conf["patch_pipe"] else lambda *args, **kwargs: None,
                style_image=style_image_path,
                style_strength=params.get("style_strength", 0.3),
                denoising_strength=params.get("denoising_strength", 0.6),
                dual_adapter_mode=params.get("dual_adapter_mode", False),
                negative_prompt=params.get("negative_prompt"),
                progress_callback=lambda step, total: log_progress(f"Generating... step {step}/{total}"),
                ip_adapter_scale=params["ips"],
                # v6: Face masking parameters
                mask_style_face=params.get("mask_style_face", True),
                face_mask_method=params.get("face_mask_method", "gaussian_blur"),
                include_hair_in_mask=params.get("include_hair_in_mask", True),
                face_mask_blur_radius=params.get("face_mask_blur_radius", 50),
                # v7: Advanced Face masking parameters (frontend control)
                mask_expand_pixels=params.get("mask_expand_pixels", 10),
                mask_edge_blur=params.get("mask_edge_blur", 10),
                controlnet_scale=params.get("controlnet_scale", 0.4),
                depth_blur_radius=params.get("depth_blur_radius", 80),
                style_strength_cap=params.get("style_strength_cap", 0.10),
                denoising_min=params.get("denoising_min", 0.90),
                # v7.2: Hair coverage ratio
                bbox_expand_ratio=params.get("bbox_expand_ratio", 1.5),
                # v7.3: Hair preservation from face reference
                hair_strength=params.get("hair_strength", 0.5),
                # v7.4: Aspect ratio adjustment toggle
                adjust_mask_aspect_ratio=params.get("adjust_mask_aspect_ratio", False),
                # v7.6: Face size matching
                match_style_face_size=params.get("match_style_face_size", True),
                # v7: Pass output directory for saving masked style image
                output_dir=OUTPUT_DIR,
            )

        # v7: Extract image and masked style path from result
        img = result["image"]
        masked_style_path = result.get("masked_style_path")
        masked_style_url = None
        if masked_style_path:
            masked_style_filename = Path(masked_style_path).name
            masked_style_url = f"/output/{masked_style_filename}"

        # Save output
        log_progress("Saving generated image...")
        output_id = str(uuid.uuid4())
        output_filename = f"{output_id}.png"
        output_path = Path(OUTPUT_DIR) / output_filename
        img.save(output_path)

        log_progress("Creating history entry...")
        # Auto-create history entry
        history = load_history()
        history_item = {
            "id": str(uuid.uuid4()),
            "title": params.get("title") or f"Generation {datetime.now().strftime('%H:%M')}",
            "created_at": datetime.now().isoformat(),
            "folder_id": None,
            "input_image_url": f"/uploads/{image_files[0].name}",
            "style_image_url": f"/uploads/{Path(style_image_path).name}" if style_image_path else None,
            "masked_style_image_url": masked_style_url,  # v7: Saved masked style image
            "output_image_url": f"/output/{output_filename}",
            "settings": {
                "model_name": params["model_name"],
                "prompt": params["prompt"],
                "negative_prompt": params.get("negative_prompt", ""),
                "ips": params["ips"],
                "lora_scale": params["lora_scale"],
                "seed": params["seed"],
                "style_strength": params.get("style_strength", 0.3),
                "denoising_strength": params.get("denoising_strength", 0.6),
                "inference_steps": params["inference_steps"],
                "dual_adapter_mode": params.get("dual_adapter_mode", False),
                "use_tiny_vae": params.get("use_tiny_vae", False),
                # v6: Face masking settings
                "mask_style_face": params.get("mask_style_face", True),
                "face_mask_method": params.get("face_mask_method", "gaussian_blur"),
                "include_hair_in_mask": params.get("include_hair_in_mask", True),
                "face_mask_blur_radius": params.get("face_mask_blur_radius", 50),
                # v7: Advanced Face masking settings
                "mask_expand_pixels": params.get("mask_expand_pixels", 10),
                "mask_edge_blur": params.get("mask_edge_blur", 10),
                "controlnet_scale": params.get("controlnet_scale", 0.4),
                "depth_blur_radius": params.get("depth_blur_radius", 80),
                "style_strength_cap": params.get("style_strength_cap", 0.10),
                "denoising_min": params.get("denoising_min", 0.90),
                # v7.4: Aspect ratio adjustment toggle
                "adjust_mask_aspect_ratio": params.get("adjust_mask_aspect_ratio", False),
                # v7.6: Face size matching
                "match_style_face_size": params.get("match_style_face_size", True),
            }
        }
        history.insert(0, history_item)
        save_history(history)

        update_task(task_id, {
            "status": TaskStatus.COMPLETED,
            "image_url": f"/output/{output_filename}",
            "masked_style_image_url": masked_style_url,  # v7: Saved masked style image
            "history_id": history_item["id"],
        })
        print(f">>> [TASK {task_id}] Completed successfully")

    except Exception as e:
        import traceback
        traceback.print_exc()
        update_task(task_id, {
            "status": TaskStatus.FAILED,
            "error": str(e)
        })
        print(f">>> [TASK {task_id}] Failed: {e}")


@app.post("/generate")
async def generate_image(
    background_tasks: BackgroundTasks,
    image_id: str = Form(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    model_name: str = Form("hyper"),
    ips: float = Form(0.8),
    lora_scale: float = Form(0.6),
    seed: int = Form(42),
    style_image_id: Optional[str] = Form(None),
    style_strength: float = Form(0.3),
    denoising_strength: float = Form(0.6),
    inference_steps: int = Form(4),
    dual_adapter_mode: bool = Form(False),
    title: str = Form(""),
    use_tiny_vae: bool = Form(False),
    # v6: Face masking parameters
    mask_style_face: bool = Form(True),
    face_mask_method: str = Form("gaussian_blur"),
    include_hair_in_mask: bool = Form(True),
    face_mask_blur_radius: int = Form(50),
    # v7: Advanced Face masking parameters (frontend control)
    mask_expand_pixels: int = Form(10),
    mask_edge_blur: int = Form(10),
    controlnet_scale: float = Form(0.4),
    depth_blur_radius: int = Form(80),
    style_strength_cap: float = Form(0.10),
    denoising_min: float = Form(0.90),
    # v7.2: Hair coverage ratio
    bbox_expand_ratio: float = Form(1.5),
    # v7.3: Hair preservation from face reference
    hair_strength: float = Form(0.5),
    # v7.4: Aspect ratio adjustment toggle
    adjust_mask_aspect_ratio: bool = Form(False),
    # v7.6: Face size matching - constrain generated face to style face size
    match_style_face_size: bool = Form(True),
):
    global generation_tasks

    # Verify image exists
    upload_path = Path(UPLOAD_DIR)
    image_files = list(upload_path.glob(f"{image_id}.*"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Uploaded image not found")

    # Create task
    task_id = str(uuid.uuid4())
    input_image_url = f"/uploads/{image_files[0].name}"

    style_image_url = None
    if style_image_id:
        style_files = list(upload_path.glob(f"{style_image_id}.*"))
        if style_files:
            style_image_url = f"/uploads/{style_files[0].name}"

    task = {
        "id": task_id,
        "status": TaskStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "input_image_url": input_image_url,
        "style_image_url": style_image_url,
        "params": {
            "image_id": image_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_name": model_name,
            "ips": ips,
            "lora_scale": lora_scale,
            "seed": seed,
            "style_image_id": style_image_id,
            "style_strength": style_strength,
            "denoising_strength": denoising_strength,
            "inference_steps": inference_steps,
            "dual_adapter_mode": dual_adapter_mode,
            "title": title,
            "use_tiny_vae": use_tiny_vae,
            # v6: Face masking parameters
            "mask_style_face": mask_style_face,
            "face_mask_method": face_mask_method,
            "include_hair_in_mask": include_hair_in_mask,
            "face_mask_blur_radius": face_mask_blur_radius,
            # v7: Advanced Face masking parameters
            "mask_expand_pixels": mask_expand_pixels,
            "mask_edge_blur": mask_edge_blur,
            "controlnet_scale": controlnet_scale,
            "depth_blur_radius": depth_blur_radius,
            "style_strength_cap": style_strength_cap,
            "denoising_min": denoising_min,
            # v7.2: Hair coverage ratio
            "bbox_expand_ratio": bbox_expand_ratio,
            # v7.3: Hair preservation from face reference
            "hair_strength": hair_strength,
            # v7.4: Aspect ratio adjustment toggle
            "adjust_mask_aspect_ratio": adjust_mask_aspect_ratio,
            # v7.6: Face size matching
            "match_style_face_size": match_style_face_size,
        },
        "image_url": None,
        "error": None,
        "history_id": None,
    }

    with task_lock:
        generation_tasks[task_id] = task
        save_tasks(generation_tasks)

    # Run generation in background thread (not asyncio - blocking operation)
    thread = threading.Thread(target=run_generation_task, args=(task_id, task["params"]))
    thread.start()

    return {
        "success": True,
        "task_id": task_id,
        "status": TaskStatus.PENDING,
    }


@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a generation task"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = generation_tasks[task_id]
    return {
        "id": task["id"],
        "status": task["status"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "input_image_url": task.get("input_image_url"),
        "style_image_url": task.get("style_image_url"),
        "image_url": task.get("image_url"),
        "error": task.get("error"),
        "history_id": task.get("history_id"),
        "params": task.get("params", {}),
        "progress_message": task.get("progress_message"),
    }


@app.get("/tasks")
async def get_active_tasks():
    """Get all active (pending/running) tasks"""
    active = []
    for task_id, task in generation_tasks.items():
        if task["status"] in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            active.append({
                "id": task["id"],
                "status": task["status"],
                "created_at": task["created_at"],
                "input_image_url": task.get("input_image_url"),
                "style_image_url": task.get("style_image_url"),
                "params": task.get("params", {}),
            })
    return {"tasks": active}


@app.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a pending task (cannot cancel running tasks)"""
    if task_id not in generation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = generation_tasks[task_id]
    if task["status"] == TaskStatus.PENDING:
        update_task(task_id, {
            "status": TaskStatus.FAILED,
            "error": "Cancelled by user"
        })
        return {"success": True}
    elif task["status"] == TaskStatus.RUNNING:
        return {"success": False, "error": "Cannot cancel running task"}
    else:
        return {"success": False, "error": "Task already completed"}


@app.delete("/upload/{file_id}")
async def delete_upload(file_id: str):
    upload_path = Path(UPLOAD_DIR)
    files = list(upload_path.glob(f"{file_id}.*"))
    for f in files:
        f.unlink()
    return {"success": True}


# ============== History API ==============

@app.get("/history")
async def get_history():
    history = load_history()
    return {"history": history}


@app.post("/history")
async def create_history(
    title: str = Body(...),
    input_image_url: str = Body(...),
    output_image_url: str = Body(...),
    model_name: str = Body(...),
    prompt: str = Body(...),
    ips: float = Body(...),
    lora_scale: float = Body(...),
    seed: int = Body(...),
    folder_id: Optional[str] = Body(None),
):
    history = load_history()

    new_item = {
        "id": str(uuid.uuid4()),
        "title": title,
        "created_at": datetime.now().isoformat(),
        "folder_id": folder_id,
        "input_image_url": input_image_url,
        "output_image_url": output_image_url,
        "settings": {
            "model_name": model_name,
            "prompt": prompt,
            "ips": ips,
            "lora_scale": lora_scale,
            "seed": seed,
        }
    }

    history.insert(0, new_item)
    save_history(history)

    return {"success": True, "item": new_item}


@app.get("/history/{history_id}")
async def get_history_item(history_id: str):
    history = load_history()
    item = next((h for h in history if h["id"] == history_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="History item not found")
    return item


@app.put("/history/{history_id}")
async def update_history(
    history_id: str,
    title: Optional[str] = Body(None),
    folder_id: Optional[str] = Body(None),
):
    history = load_history()
    item_index = next((i for i, h in enumerate(history) if h["id"] == history_id), None)

    if item_index is None:
        raise HTTPException(status_code=404, detail="History item not found")

    if title is not None:
        history[item_index]["title"] = title
    if folder_id is not None:
        history[item_index]["folder_id"] = folder_id if folder_id != "" else None

    save_history(history)
    return {"success": True, "item": history[item_index]}


@app.delete("/history/{history_id}")
async def delete_history(history_id: str):
    history = load_history()
    item = next((h for h in history if h["id"] == history_id), None)

    if not item:
        raise HTTPException(status_code=404, detail="History item not found")

    history = [h for h in history if h["id"] != history_id]
    save_history(history)

    return {"success": True}


# ============== Folders API ==============

@app.get("/folders")
async def get_folders():
    folders = load_folders()
    return {"folders": folders}


@app.post("/folders")
async def create_folder(name: str = Body(..., embed=True)):
    folders = load_folders()

    new_folder = {
        "id": str(uuid.uuid4()),
        "name": name,
        "created_at": datetime.now().isoformat(),
        "order": len(folders),
    }

    folders.append(new_folder)
    save_folders(folders)

    return {"success": True, "folder": new_folder}


@app.put("/folders/{folder_id}")
async def update_folder(
    folder_id: str,
    name: Optional[str] = Body(None),
    order: Optional[int] = Body(None),
):
    folders = load_folders()
    folder_index = next((i for i, f in enumerate(folders) if f["id"] == folder_id), None)

    if folder_index is None:
        raise HTTPException(status_code=404, detail="Folder not found")

    if name is not None:
        folders[folder_index]["name"] = name
    if order is not None:
        folders[folder_index]["order"] = order

    save_folders(folders)
    return {"success": True, "folder": folders[folder_index]}


@app.delete("/folders/{folder_id}")
async def delete_folder(folder_id: str):
    folders = load_folders()
    folder = next((f for f in folders if f["id"] == folder_id), None)

    if not folder:
        raise HTTPException(status_code=404, detail="Folder not found")

    # Move items in this folder to no folder
    history = load_history()
    for item in history:
        if item.get("folder_id") == folder_id:
            item["folder_id"] = None
    save_history(history)

    folders = [f for f in folders if f["id"] != folder_id]
    save_folders(folders)

    return {"success": True}


# ============== Mask Preview API ==============

@app.post("/preview-mask")
async def preview_mask(
    image_id: str = Form(...),
    image_type: str = Form(...),  # "face", "style", or "hair"
    include_hair_in_mask: bool = Form(True),
    mask_expand_pixels: int = Form(10),
    mask_edge_blur: int = Form(10),
    face_mask_method: str = Form("gaussian_blur"),  # gaussian_blur, fill, noise
    face_mask_blur_radius: int = Form(50),
    bbox_expand_ratio: float = Form(1.5),  # v7.2: Hair coverage ratio (1.0-3.0)
    face_image_id: Optional[str] = Form(None),  # v7.3: For aspect ratio adjustment
    adjust_mask_aspect_ratio: bool = Form(False),  # v7.4: Enable/disable aspect ratio adjustment
):
    """
    Generate mask preview for visualization.
    - For face image: returns face detection bounding box overlay
    - For style image: returns face mask overlay
    """
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    import cv2
    import io
    import base64

    upload_path = Path(UPLOAD_DIR)
    image_files = list(upload_path.glob(f"{image_id}.*"))
    if not image_files:
        raise HTTPException(status_code=404, detail="Image not found")

    image_path = str(image_files[0])

    try:
        if image_type == "face":
            # Face detection - show bounding box
            from insightface.app import FaceAnalysis

            # Initialize InsightFace if needed
            app_face = FaceAnalysis(
                name="buffalo_l",
                root="models_cache",
                providers=["CPUExecutionProvider"]
            )
            app_face.prepare(ctx_id=0, det_size=(640, 640))

            # Load and detect
            img_pil = Image.open(image_path).convert("RGB")
            img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            faces = app_face.get(img_cv)

            if len(faces) == 0:
                return {
                    "success": False,
                    "error": "No face detected",
                    "overlay_url": None
                }

            # Create overlay with face bounding box
            overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Draw semi-transparent rectangle
                draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    outline=(0, 200, 255, 255),
                    width=3
                )

                # Fill with semi-transparent color
                fill_overlay = Image.new("RGBA", img_pil.size, (0, 0, 0, 0))
                fill_draw = ImageDraw.Draw(fill_overlay)
                fill_draw.rectangle(
                    [(x1, y1), (x2, y2)],
                    fill=(0, 200, 255, 60)
                )
                overlay = Image.alpha_composite(overlay, fill_overlay)

                # Draw facial landmarks if available
                if hasattr(face, 'kps') and face.kps is not None:
                    for kp in face.kps:
                        x, y = int(kp[0]), int(kp[1])
                        draw.ellipse(
                            [(x - 3, y - 3), (x + 3, y + 3)],
                            fill=(255, 100, 100, 255)
                        )

            # Calculate aspect ratio for display
            face = faces[0]
            bbox = face.bbox
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            aspect_ratio = height / width if width > 0 else 1.0

            # Save overlay to output
            overlay_id = str(uuid.uuid4())
            overlay_filename = f"mask_preview_{overlay_id}.png"
            overlay_path = Path(OUTPUT_DIR) / overlay_filename
            overlay.save(overlay_path)

            return {
                "success": True,
                "overlay_url": f"/output/{overlay_filename}",
                "face_detected": True,
                "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                "aspect_ratio": float(round(aspect_ratio, 2))
            }

        elif image_type == "style":
            # Style image - show face mask
            from src.face_parsing import FaceParser
            from insightface.app import FaceAnalysis

            device = get_device()
            face_parser = FaceParser(device=str(device))

            img_pil = Image.open(image_path).convert("RGB")
            original_size = img_pil.size

            # v7.1: Detect face bbox using InsightFace to filter SegFormer noise
            style_face_bbox = None
            try:
                app_face = FaceAnalysis(
                    name="buffalo_l",
                    root="models_cache",
                    providers=["CPUExecutionProvider"]
                )
                app_face.prepare(ctx_id=0, det_size=(640, 640))
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                faces = app_face.get(img_cv)
                if len(faces) > 0:
                    bbox = faces[0].bbox
                    style_face_bbox = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                    print(f">>> [preview-mask] Style face bbox: {style_face_bbox}")
            except Exception as e:
                print(f">>> [preview-mask] Face detection failed: {e}")

            # Use 1024x1024 for mask generation (same as actual generation pipeline)
            # SegFormer works better at this resolution
            GENERATION_SIZE = (1024, 1024)

            # Get face mask at generation resolution
            mask = face_parser.get_face_hair_mask(
                img_pil,
                target_size=GENERATION_SIZE,
                include_hair=include_hair_in_mask,
                expand_pixels=mask_expand_pixels,
                blur_radius=mask_edge_blur,
                face_bbox=style_face_bbox,  # v7.1: Pass bbox to filter noise
                bbox_expand_ratio=bbox_expand_ratio  # v7.2: Frontend controlled
            )

            if mask is None:
                return {
                    "success": False,
                    "error": "No face detected in style image",
                    "overlay_url": None
                }

            # v7.3: Adjust mask aspect ratio to match input face (same as generation pipeline)
            input_face_aspect_ratio = None
            if face_image_id:
                try:
                    face_files = list(upload_path.glob(f"{face_image_id}.*"))
                    if face_files:
                        face_img = Image.open(str(face_files[0])).convert("RGB")
                        face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
                        face_faces = app_face.get(face_cv)
                        if len(face_faces) > 0:
                            face_bbox = face_faces[0].bbox
                            face_w = float(face_bbox[2] - face_bbox[0])
                            face_h = float(face_bbox[3] - face_bbox[1])
                            input_face_aspect_ratio = face_h / face_w if face_w > 0 else 1.0
                            print(f">>> [preview-mask] Input face aspect ratio: {input_face_aspect_ratio:.2f}")
                except Exception as e:
                    print(f">>> [preview-mask] Failed to get input face aspect ratio: {e}")

            # v7.4: Only apply aspect ratio adjustment if toggle is enabled
            if adjust_mask_aspect_ratio and input_face_aspect_ratio is not None:
                mask = face_parser.adjust_mask_for_aspect_ratio(
                    mask,
                    target_aspect_ratio=input_face_aspect_ratio
                )
                print(f">>> [preview-mask] Mask adjusted for aspect ratio")
            elif input_face_aspect_ratio is not None:
                print(f">>> [preview-mask] Aspect ratio adjustment disabled (toggle off)")

            # Apply actual mask to style image (same as generation pipeline)
            # Resize style image to generation size for consistent masking
            style_resized = img_pil.resize(GENERATION_SIZE, Image.LANCZOS)

            # Apply mask using same method as actual generation
            masked_style = face_parser.apply_mask(
                style_resized,
                mask,
                method=face_mask_method,
                blur_radius=face_mask_blur_radius
            )

            # Resize masked result and mask back to original size
            masked_style_display = masked_style.resize(original_size, Image.LANCZOS)
            mask_resized = mask.resize(original_size, Image.LANCZOS)

            # Create overlay: show blurred region only, rest transparent
            # This clearly shows WHERE the blur is applied
            masked_rgba = masked_style_display.convert("RGBA")
            masked_array = np.array(masked_rgba)
            mask_array_resized = np.array(mask_resized)

            # Set alpha based on mask: boost alpha so soft edges are visible
            # The mask has soft edges from GaussianBlur, but we want the full extent visible
            # Amplify values and set minimum alpha for any masked area
            boosted_alpha = np.clip(mask_array_resized.astype(np.float32) * 1.5 + 30, 0, 255).astype(np.uint8)
            boosted_alpha[mask_array_resized < 10] = 0  # Keep truly transparent areas transparent
            masked_array[:, :, 3] = boosted_alpha

            overlay = Image.fromarray(masked_array, mode="RGBA")

            # Get mask info for display
            mask_array = np.array(mask)

            # Get mask bbox for info
            bbox = face_parser.get_mask_bbox(mask)
            aspect_ratio = face_parser.get_face_aspect_ratio(bbox) if bbox else None

            # Save overlay
            overlay_id = str(uuid.uuid4())
            overlay_filename = f"mask_preview_{overlay_id}.png"
            overlay_path = Path(OUTPUT_DIR) / overlay_filename
            overlay.save(overlay_path)

            # Cleanup
            face_parser.unload()

            # Convert numpy types to native Python types for JSON serialization
            mask_coverage = float(round(np.sum(mask_array > 127) / mask_array.size * 100, 1))
            bbox_list = [int(x) for x in bbox] if bbox else None
            aspect_ratio_val = float(round(aspect_ratio, 2)) if aspect_ratio else None

            return {
                "success": True,
                "overlay_url": f"/output/{overlay_filename}",
                "mask_coverage": mask_coverage,
                "bbox": bbox_list,
                "aspect_ratio": aspect_ratio_val
            }

        elif image_type == "hair":
            # v7.3: Hair region extraction preview from face reference
            from src.face_parsing import FaceParser, extract_hair_region

            device = get_device()
            face_parser = FaceParser(device=str(device))

            img_pil = Image.open(image_path).convert("RGB")
            original_size = img_pil.size

            # Extract hair region
            hair_region = extract_hair_region(img_pil, face_parser)

            if hair_region is None:
                face_parser.unload()
                return {
                    "success": False,
                    "error": "No hair detected in face image",
                    "overlay_url": None,
                    "hair_coverage": 0
                }

            # Get segmentation to calculate hair coverage
            seg_map = face_parser.get_segmentation(img_pil, target_size=original_size)
            hair_mask = np.zeros(seg_map.shape, dtype=np.uint8) if seg_map is not None else None
            if seg_map is not None:
                hair_mask[seg_map == 17] = 255

            hair_coverage = float(round(np.sum(hair_mask > 0) / hair_mask.size * 100, 1)) if hair_mask is not None else 0

            # Create overlay: hair region with semi-transparent highlight
            # Show extracted hair on top of original with color tint
            hair_rgba = hair_region.convert("RGBA")
            hair_array = np.array(hair_rgba)

            # Create highlight overlay for hair region
            overlay = Image.new("RGBA", original_size, (0, 0, 0, 0))
            if hair_mask is not None:
                # Create colored overlay where hair is detected
                overlay_array = np.zeros((original_size[1], original_size[0], 4), dtype=np.uint8)
                # Purple tint for hair region
                overlay_array[:, :, 0] = 180  # R
                overlay_array[:, :, 1] = 100  # G
                overlay_array[:, :, 2] = 220  # B
                overlay_array[:, :, 3] = (hair_mask * 0.6).astype(np.uint8)  # Alpha based on mask
                overlay = Image.fromarray(overlay_array, mode="RGBA")

            # Save overlay
            overlay_id = str(uuid.uuid4())
            overlay_filename = f"hair_preview_{overlay_id}.png"
            overlay_path = Path(OUTPUT_DIR) / overlay_filename
            overlay.save(overlay_path)

            # Also save the extracted hair region for reference
            hair_region_filename = f"hair_region_{overlay_id}.png"
            hair_region_path = Path(OUTPUT_DIR) / hair_region_filename
            hair_region.save(hair_region_path)

            face_parser.unload()

            return {
                "success": True,
                "overlay_url": f"/output/{overlay_filename}",
                "hair_region_url": f"/output/{hair_region_filename}",
                "hair_coverage": hair_coverage,
                "hair_detected": True
            }

        else:
            raise HTTPException(status_code=400, detail="Invalid image_type. Use 'face', 'style', or 'hair'")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ============== VLM Prompt Generation ==============

@app.post("/generate-prompt")
async def generate_prompt_from_images(
    face_image_id: str = Form(...),
    style_image_id: Optional[str] = Form(None),
):
    """Generate prompts using Gemini VLM based on uploaded images"""
    import base64
    import httpx

    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not configured")

    upload_path = Path(UPLOAD_DIR)

    # Find face image
    face_files = list(upload_path.glob(f"{face_image_id}.*"))
    if not face_files:
        raise HTTPException(status_code=404, detail="Face image not found")

    # Read and encode face image
    with open(face_files[0], "rb") as f:
        face_image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    face_mime = f"image/{face_files[0].suffix[1:].lower()}"
    if face_mime == "image/jpg":
        face_mime = "image/jpeg"

    # Prepare image parts for Gemini
    image_parts = [
        {
            "inline_data": {
                "mime_type": face_mime,
                "data": face_image_data
            }
        }
    ]

    # Add style image if provided
    style_description = ""
    if style_image_id:
        style_files = list(upload_path.glob(f"{style_image_id}.*"))
        if style_files:
            with open(style_files[0], "rb") as f:
                style_image_data = base64.standard_b64encode(f.read()).decode("utf-8")
            style_mime = f"image/{style_files[0].suffix[1:].lower()}"
            if style_mime == "image/jpg":
                style_mime = "image/jpeg"
            image_parts.append({
                "inline_data": {
                    "mime_type": style_mime,
                    "data": style_image_data
                }
            })
            style_description = "The second image is a STYLE REFERENCE."

    # Construct the prompt - very strict to prevent hallucination
    style_instruction = ""
    if style_image_id:
        style_instruction = """- Second image is STYLE REFERENCE. For background:
  * If solid color: "solid [basic color name] background" (use only: white, black, gray, red, blue, green, yellow, orange, pink, purple, brown, beige)
  * If has objects: list ONLY clearly visible objects, no interpretation
  * IGNORE any text, logos, watermarks in the style image"""

    system_prompt = f"""Create a Stable Diffusion prompt. Be LITERAL and MINIMAL.

IMAGE 1: Person photo
{style_instruction if style_image_id else ""}

OUTPUT FORMAT - JSON only:
{{"positive": "[description]", "negative": "ugly, deformed, blurry, low quality, bad anatomy"}}

POSITIVE PROMPT RULES:
- Person: gender, age range (young/middle-aged/elderly), visible clothing color/type
- Background: {"Use ONLY the exact color you see. Example: 'solid dark green background' or 'solid gray background'. Do NOT add texture words." if style_image_id else "'solid white background' or 'studio background'"}
- Lighting: "soft lighting" or "natural lighting" ONLY
- Maximum 25 words total
- Write in simple noun phrases, no adjectives beyond color

BANNED WORDS (never use these):
elegant, sophisticated, painterly, artistic, subtle, gentle, serene, dramatic, rich, vibrant, warm, cool, moody, atmospheric, textured, professional, high-end, luxurious, minimalist, modern, classic, vintage

If you're unsure about ANY detail, leave it out."""

    # Call Gemini API
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{
            "parts": image_parts + [{"text": system_prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "topK": 20,
            "topP": 0.8,
            "maxOutputTokens": 512,
        }
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

        # Extract text from response
        text = result["candidates"][0]["content"]["parts"][0]["text"]

        # Parse JSON from response (handle markdown code blocks)
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        prompts = json.loads(text.strip())

        return {
            "success": True,
            "positive": prompts.get("positive", ""),
            "negative": prompts.get("negative", "")
        }

    except httpx.HTTPStatusError as e:
        print(f"Gemini API error: {e.response.text}")
        raise HTTPException(status_code=500, detail=f"Gemini API error: {e.response.status_code}")
    except json.JSONDecodeError as e:
        print(f"Failed to parse Gemini response: {text}")
        raise HTTPException(status_code=500, detail="Failed to parse VLM response")
    except Exception as e:
        print(f"Error generating prompt: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== PDF Report Generation ==============

@app.post("/generate-report")
async def generate_pdf_report(
    face_image_id: str = Form(...),
    style_image_id: Optional[str] = Form(None),
    result_image_url: str = Form(...),
    # Generation parameters
    model_name: str = Form("hyper"),
    ips: float = Form(0.6),
    lora_scale: float = Form(1.0),
    style_strength: float = Form(0.3),
    denoising_strength: float = Form(0.7),
    inference_steps: int = Form(4),
    seed: int = Form(-1),
    use_tiny_vae: bool = Form(False),
    positive_prompt: str = Form(""),
    negative_prompt: str = Form(""),
    # Face masking parameters
    mask_style_face: bool = Form(False),
    include_hair_in_mask: bool = Form(True),
    face_mask_method: str = Form("gaussian_blur"),
    face_mask_blur_radius: int = Form(30),
    mask_expand_pixels: int = Form(10),
    mask_edge_blur: int = Form(10),
    controlnet_scale: float = Form(0.4),
    depth_blur_radius: int = Form(50),
    style_strength_cap: float = Form(0.2),
    denoising_min: float = Form(0.75),
    adjust_mask_aspect_ratio: bool = Form(False),
):
    """Generate a PDF report with images and all parameters used"""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from io import BytesIO
    from datetime import datetime
    import tempfile

    # Try to register Korean font if available
    try:
        # Mac system font path
        font_paths = [
            "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/Library/Fonts/NanumGothic.ttf",
        ]
        font_registered = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                pdfmetrics.registerFont(TTFont('Korean', font_path))
                font_registered = True
                break
    except:
        font_registered = False

    upload_path = Path(UPLOAD_DIR)
    output_path = Path(OUTPUT_DIR)

    # Create PDF buffer
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=20*mm, leftMargin=20*mm, topMargin=20*mm, bottomMargin=20*mm)

    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        spaceAfter=12,
        fontName='Korean' if font_registered else 'Helvetica-Bold'
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=12,
        spaceAfter=6,
        spaceBefore=12,
        fontName='Korean' if font_registered else 'Helvetica-Bold'
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        fontName='Korean' if font_registered else 'Helvetica'
    )

    elements = []

    # Title
    elements.append(Paragraph("FastFace Generation Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 10*mm))

    # Images section
    elements.append(Paragraph("Images", heading_style))

    # Prepare images for the table
    img_data = []
    img_labels = []

    # Face image
    face_files = list(upload_path.glob(f"{face_image_id}.*"))
    if face_files:
        face_img = RLImage(str(face_files[0]), width=50*mm, height=50*mm)
        img_data.append(face_img)
        img_labels.append("Face Input")

    # Style image
    if style_image_id:
        style_files = list(upload_path.glob(f"{style_image_id}.*"))
        if style_files:
            style_img = RLImage(str(style_files[0]), width=50*mm, height=50*mm)
            img_data.append(style_img)
            img_labels.append("Style Input")

    # Result image
    result_filename = result_image_url.split("/")[-1]
    result_file = output_path / result_filename
    if result_file.exists():
        result_img = RLImage(str(result_file), width=50*mm, height=50*mm)
        img_data.append(result_img)
        img_labels.append("Generated Result")

    # Create image table
    if img_data:
        img_table = Table([img_data, img_labels], colWidths=[55*mm] * len(img_data))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 1), (-1, 1), 'Korean' if font_registered else 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, 1), 8),
            ('TOPPADDING', (0, 1), (-1, 1), 5),
        ]))
        elements.append(img_table)

    elements.append(Spacer(1, 8*mm))

    # Parameters section
    elements.append(Paragraph("Generation Parameters", heading_style))

    # Basic parameters table
    basic_params = [
        ["Parameter", "Value"],
        ["Model", model_name],
        ["ID Similarity (ips)", f"{ips:.2f}"],
        ["LoRA Scale", f"{lora_scale:.2f}"],
        ["Style Strength", f"{style_strength:.2f}"],
        ["Denoising Strength", f"{denoising_strength:.2f}"],
        ["Inference Steps", str(inference_steps)],
        ["Seed", str(seed)],
        ["VAE", "Tiny" if use_tiny_vae else "Full"],
    ]

    param_table = Table(basic_params, colWidths=[60*mm, 80*mm])
    param_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.95)]),
    ]))
    elements.append(param_table)

    # Prompts section
    if positive_prompt or negative_prompt:
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("Prompts", heading_style))

        prompt_data = [["Type", "Content"]]
        if positive_prompt:
            # Truncate long prompts
            pos_display = positive_prompt[:200] + "..." if len(positive_prompt) > 200 else positive_prompt
            prompt_data.append(["Positive", pos_display])
        if negative_prompt:
            neg_display = negative_prompt[:200] + "..." if len(negative_prompt) > 200 else negative_prompt
            prompt_data.append(["Negative", neg_display])

        prompt_table = Table(prompt_data, colWidths=[30*mm, 110*mm])
        prompt_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        elements.append(prompt_table)

    # Face Masking section (if enabled)
    if mask_style_face:
        elements.append(Spacer(1, 5*mm))
        elements.append(Paragraph("Face Masking Settings", heading_style))

        mask_params = [
            ["Parameter", "Value"],
            ["Face Masking", "Enabled"],
            ["Include Hair", "Yes" if include_hair_in_mask else "No"],
            ["Mask Method", face_mask_method],
            ["Blur Radius", str(face_mask_blur_radius)],
            ["Mask Expand", f"{mask_expand_pixels}px"],
            ["Edge Blur", str(mask_edge_blur)],
            ["ControlNet Scale", f"{controlnet_scale:.2f}"],
            ["Depth Blur", str(depth_blur_radius)],
            ["Style Cap", f"{style_strength_cap:.2f}"],
            ["Denoising Min", f"{denoising_min:.2f}"],
            ["Aspect Ratio Adjust", "Yes" if adjust_mask_aspect_ratio else "No"],
        ]

        mask_table = Table(mask_params, colWidths=[60*mm, 80*mm])
        mask_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.3, 0.3, 0.5)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.Color(0.95, 0.95, 0.98)]),
        ]))
        elements.append(mask_table)

    # Build PDF
    doc.build(elements)

    # Save to output directory
    report_id = str(uuid.uuid4())[:8]
    report_filename = f"report_{report_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    report_path = output_path / report_filename

    with open(report_path, 'wb') as f:
        f.write(buffer.getvalue())

    return {
        "success": True,
        "report_url": f"/output/{report_filename}",
        "filename": report_filename
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
