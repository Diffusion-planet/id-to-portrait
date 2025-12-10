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
DEVICE = os.getenv("DEVICE", "mps")
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
    "hyper": [1, 2, 4, 8],
    "realvis": [1, 2, 4, 8],
    "lightning": [2, 4, 8],
    "turbo": [1, 2, 4],  # turbo is fast, doesn't need many steps
    "lcm": [2, 4, 6, 8],
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
        if params.get("style_image_id"):
            style_files = list(upload_path.glob(f"{params['style_image_id']}.*"))
            if style_files:
                style_image_path = str(style_files[0])
                log_progress("Style image found, preparing dual adapter mode...")

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

        if conf["patch_pipe"]:
            log_progress("Applying attention manipulation patches...")
            patch_pipe(pipe, **conf["am_patch_kwargs"])

        generator = torch.Generator(device=device).manual_seed(params["seed"])

        log_progress("Extracting face embedding...")
        log_progress(f"[DEBUG] dual_adapter_mode={params.get('dual_adapter_mode')}, style_image_path={style_image_path}")

        with torch.no_grad():
            img = pipe.execute(
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
            )

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
            }
        }
        history.insert(0, history_item)
        save_history(history)

        update_task(task_id, {
            "status": TaskStatus.COMPLETED,
            "image_url": f"/output/{output_filename}",
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
