import os
import sys
import uuid
import json
import gc
import shutil
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
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

# Global pipeline instance
pipeline = None
current_model = None

# Create directories on import
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs("models_cache", exist_ok=True)


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


def load_pipeline(model_name: str):
    global pipeline, current_model

    if pipeline is not None and current_model == model_name:
        return pipeline

    # Clean up existing pipeline
    if pipeline is not None:
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

    with open(config_path, "r") as f:
        conf = json.load(f)

    conf["model_name"] = model_name
    conf["pipe_kwargs"]["num_inference_steps"] = 4

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

    pipe = sdxl_pipeline.name2pipe["faceid"](conf, device)

    # Use lightweight VAE
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesdxl",
        torch_dtype=torch.float16,
        cache_dir="models_cache/"
    ).to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    pipe._config = conf
    pipeline = pipe
    current_model = model_name

    return pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("models_cache", exist_ok=True)
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
            {"id": "hyper", "name": "Hyper-SD", "description": "4-step fast generation", "default": True},
            {"id": "lightning", "name": "SDXL-Lightning", "description": "4-step lightning model"},
            {"id": "lcm", "name": "LCM", "description": "Latent Consistency Model"},
            {"id": "turbo", "name": "SDXL-Turbo", "description": "512x512 fast generation"},
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


@app.post("/generate", response_model=InferenceResponse)
async def generate_image(
    image_id: str = Form(...),
    prompt: str = Form(...),
    model_name: str = Form("hyper"),
    ips: float = Form(0.8),
    lora_scale: float = Form(0.6),
    seed: int = Form(42),
    style_image_id: Optional[str] = Form(None),
    style_strength: float = Form(0.3),
):
    from src.custom_dca import patch_pipe, reset_patched_unet

    try:
        # Find uploaded face image
        upload_path = Path(UPLOAD_DIR)
        image_files = list(upload_path.glob(f"{image_id}.*"))
        if not image_files:
            raise HTTPException(status_code=404, detail="Uploaded image not found")

        image_path = str(image_files[0])

        # Find style image if provided
        style_image_path = None
        if style_image_id:
            style_files = list(upload_path.glob(f"{style_image_id}.*"))
            if style_files:
                style_image_path = str(style_files[0])

        # Load pipeline
        pipe = load_pipeline(model_name)
        conf = pipe._config
        device = get_device()

        # Set parameters
        if conf["method"] == "faceid" and "faceid_lora_scale" in conf:
            if "faceid_0" in pipe.unet.peft_config:
                pipe.set_adapters(["faceid_0"], lora_scale)

        pipe.set_ip_adapter_scale(ips)

        # Apply patches
        if conf["patch_pipe"]:
            patch_pipe(pipe, **conf["am_patch_kwargs"])

        # Generate
        generator = torch.Generator(device=device).manual_seed(seed)

        print(f">>> [DEBUG] face_image_path: {image_path}")
        print(f">>> [DEBUG] style_image_path: {style_image_path}")
        print(f">>> [DEBUG] prompt: {prompt}")

        with torch.no_grad():
            img = pipe.execute(
                image_path,
                prompt,
                generator,
                conf["pipe_kwargs"],
                after_hook_fn=reset_patched_unet if conf["patch_pipe"] else lambda *args, **kwargs: None,
                style_image=style_image_path,
                style_strength=style_strength,
            )

        # Save output
        output_id = str(uuid.uuid4())
        output_filename = f"{output_id}.png"
        output_path = Path(OUTPUT_DIR) / output_filename
        img.save(output_path)

        return InferenceResponse(
            success=True,
            image_url=f"/output/{output_filename}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return InferenceResponse(
            success=False,
            error=str(e)
        )


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
