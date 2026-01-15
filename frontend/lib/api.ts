const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8007'

export interface Model {
  id: string
  name: string
  description: string
  default?: boolean
  valid_steps?: number[]
  default_steps?: number
}

export interface UploadResponse {
  success: boolean
  file_id: string
  filename: string
  url: string
}

export type FaceMaskMethod = 'gaussian_blur' | 'fill' | 'noise'

export interface GenerateParams {
  image_id: string
  prompt: string
  negative_prompt?: string
  model_name: string
  ips: number
  lora_scale: number
  seed: number
  style_image_id?: string | null
  style_strength?: number
  denoising_strength?: number
  inference_steps?: number
  title?: string
  use_tiny_vae?: boolean
  // v6 Face Masking Parameters
  mask_style_face?: boolean
  face_mask_method?: FaceMaskMethod
  include_hair_in_mask?: boolean
  face_mask_blur_radius?: number
  // v7: Advanced Face Masking Parameters (frontend control)
  mask_expand_pixels?: number
  mask_edge_blur?: number
  controlnet_scale?: number
  depth_blur_radius?: number
  style_strength_cap?: number
  denoising_min?: number
  // v7.2: Hair coverage ratio
  bbox_expand_ratio?: number
  // v7.3: Hair preservation from face reference
  hair_strength?: number
  // v7.4: Aspect ratio adjustment toggle
  adjust_mask_aspect_ratio?: boolean
  // v7.6: Face size matching
  match_style_face_size?: boolean
}

export interface GenerateResponse {
  success: boolean
  task_id: string
  status: TaskStatus
}

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed'

export interface Task {
  id: string
  status: TaskStatus
  created_at: string
  updated_at: string
  input_image_url?: string
  style_image_url?: string
  masked_style_image_url?: string  // v7: Saved masked style image from generation
  image_url?: string
  error?: string
  history_id?: string
  params: GenerateParams
  progress_message?: string
}

export interface HealthResponse {
  status: string
  device: string
  model_loaded: boolean
  current_model: string | null
}

export interface HistorySettings {
  model_name: string
  prompt: string
  negative_prompt?: string
  ips: number
  lora_scale: number
  seed: number
  style_strength?: number
  denoising_strength?: number
  inference_steps?: number
  use_tiny_vae?: boolean
  // v6 Face Masking Parameters
  mask_style_face?: boolean
  face_mask_method?: FaceMaskMethod
  include_hair_in_mask?: boolean
  face_mask_blur_radius?: number
  // v7: Advanced Face Masking Parameters
  mask_expand_pixels?: number
  mask_edge_blur?: number
  controlnet_scale?: number
  depth_blur_radius?: number
  style_strength_cap?: number
  denoising_min?: number
  // v7.2: Hair coverage ratio
  bbox_expand_ratio?: number
  // v7.3: Hair preservation from face reference
  hair_strength?: number
  // v7.4: Aspect ratio adjustment toggle
  adjust_mask_aspect_ratio?: boolean
  // v7.6: Face size matching
  match_style_face_size?: boolean
}

export interface HistoryItem {
  id: string
  title: string
  created_at: string
  folder_id: string | null
  input_image_url: string
  style_image_url?: string | null
  masked_style_image_url?: string | null  // v7: Saved masked style image from generation
  output_image_url: string
  settings: HistorySettings
}

export interface Folder {
  id: string
  name: string
  created_at: string
  order: number
}

export async function checkHealth(): Promise<HealthResponse> {
  const res = await fetch(`${API_URL}/health`)
  if (!res.ok) throw new Error('API not available')
  return res.json()
}

export async function getModels(): Promise<{ models: Model[] }> {
  const res = await fetch(`${API_URL}/models`)
  if (!res.ok) throw new Error('Failed to fetch models')
  return res.json()
}

export async function uploadImage(file: File): Promise<UploadResponse> {
  const formData = new FormData()
  formData.append('file', file)

  const res = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) throw new Error('Failed to upload image')
  return res.json()
}

export async function generateImage(params: GenerateParams): Promise<GenerateResponse> {
  const formData = new FormData()
  formData.append('image_id', params.image_id)
  formData.append('prompt', params.prompt)
  if (params.negative_prompt) {
    formData.append('negative_prompt', params.negative_prompt)
  }
  formData.append('model_name', params.model_name)
  formData.append('ips', params.ips.toString())
  formData.append('lora_scale', params.lora_scale.toString())
  formData.append('seed', params.seed.toString())

  // Add optional style image parameters
  if (params.style_image_id) {
    formData.append('style_image_id', params.style_image_id)
  }
  if (params.style_strength !== undefined) {
    formData.append('style_strength', params.style_strength.toString())
  }
  if (params.denoising_strength !== undefined) {
    formData.append('denoising_strength', params.denoising_strength.toString())
  }
  if (params.inference_steps !== undefined) {
    formData.append('inference_steps', params.inference_steps.toString())
  }
  if (params.dual_adapter_mode !== undefined) {
    formData.append('dual_adapter_mode', params.dual_adapter_mode.toString())
  }
  if (params.title) {
    formData.append('title', params.title)
  }
  if (params.use_tiny_vae !== undefined) {
    formData.append('use_tiny_vae', params.use_tiny_vae.toString())
  }

  // v6 Face Masking Parameters
  if (params.mask_style_face !== undefined) {
    formData.append('mask_style_face', params.mask_style_face.toString())
  }
  if (params.face_mask_method) {
    formData.append('face_mask_method', params.face_mask_method)
  }
  if (params.include_hair_in_mask !== undefined) {
    formData.append('include_hair_in_mask', params.include_hair_in_mask.toString())
  }
  if (params.face_mask_blur_radius !== undefined) {
    formData.append('face_mask_blur_radius', params.face_mask_blur_radius.toString())
  }

  // v7: Advanced Face Masking Parameters (frontend control)
  if (params.mask_expand_pixels !== undefined) {
    formData.append('mask_expand_pixels', params.mask_expand_pixels.toString())
  }
  if (params.mask_edge_blur !== undefined) {
    formData.append('mask_edge_blur', params.mask_edge_blur.toString())
  }
  if (params.controlnet_scale !== undefined) {
    formData.append('controlnet_scale', params.controlnet_scale.toString())
  }
  if (params.depth_blur_radius !== undefined) {
    formData.append('depth_blur_radius', params.depth_blur_radius.toString())
  }
  if (params.style_strength_cap !== undefined) {
    formData.append('style_strength_cap', params.style_strength_cap.toString())
  }
  if (params.denoising_min !== undefined) {
    formData.append('denoising_min', params.denoising_min.toString())
  }
  // v7.2: Hair coverage ratio
  if (params.bbox_expand_ratio !== undefined) {
    formData.append('bbox_expand_ratio', params.bbox_expand_ratio.toString())
  }
  // v7.3: Hair preservation from face reference
  if (params.hair_strength !== undefined) {
    formData.append('hair_strength', params.hair_strength.toString())
  }
  // v7.4: Aspect ratio adjustment toggle
  if (params.adjust_mask_aspect_ratio !== undefined) {
    formData.append('adjust_mask_aspect_ratio', params.adjust_mask_aspect_ratio.toString())
  }
  // v7.6: Face size matching
  if (params.match_style_face_size !== undefined) {
    formData.append('match_style_face_size', params.match_style_face_size.toString())
  }

  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) throw new Error('Failed to generate image')
  return res.json()
}

// ============== Tasks API ==============

export async function getTaskStatus(taskId: string): Promise<Task> {
  const res = await fetch(`${API_URL}/tasks/${taskId}`)
  if (!res.ok) throw new Error('Task not found')
  return res.json()
}

export async function getActiveTasks(): Promise<{ tasks: Task[] }> {
  const res = await fetch(`${API_URL}/tasks`)
  if (!res.ok) throw new Error('Failed to fetch tasks')
  return res.json()
}

export async function cancelTask(taskId: string): Promise<{ success: boolean; error?: string }> {
  const res = await fetch(`${API_URL}/tasks/${taskId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to cancel task')
  return res.json()
}

export function getImageUrl(path: string): string {
  if (path.startsWith('http')) return path
  return `${API_URL}${path}`
}

// ============== History API ==============

export async function getHistory(): Promise<{ history: HistoryItem[] }> {
  const res = await fetch(`${API_URL}/history`)
  if (!res.ok) throw new Error('Failed to fetch history')
  return res.json()
}

export async function createHistory(params: {
  title: string
  input_image_url: string
  output_image_url: string
  model_name: string
  prompt: string
  ips: number
  lora_scale: number
  seed: number
  folder_id?: string | null
}): Promise<{ success: boolean; item: HistoryItem }> {
  const res = await fetch(`${API_URL}/history`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Failed to create history')
  return res.json()
}

export async function updateHistory(
  historyId: string,
  params: { title?: string; folder_id?: string | null }
): Promise<{ success: boolean; item: HistoryItem }> {
  const res = await fetch(`${API_URL}/history/${historyId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Failed to update history')
  return res.json()
}

export async function deleteHistory(historyId: string): Promise<{ success: boolean }> {
  const res = await fetch(`${API_URL}/history/${historyId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete history')
  return res.json()
}

// ============== Folders API ==============

export async function getFolders(): Promise<{ folders: Folder[] }> {
  const res = await fetch(`${API_URL}/folders`)
  if (!res.ok) throw new Error('Failed to fetch folders')
  return res.json()
}

export async function createFolder(name: string): Promise<{ success: boolean; folder: Folder }> {
  const res = await fetch(`${API_URL}/folders`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error('Failed to create folder')
  return res.json()
}

export async function updateFolder(
  folderId: string,
  params: { name?: string; order?: number }
): Promise<{ success: boolean; folder: Folder }> {
  const res = await fetch(`${API_URL}/folders/${folderId}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })
  if (!res.ok) throw new Error('Failed to update folder')
  return res.json()
}

export async function deleteFolder(folderId: string): Promise<{ success: boolean }> {
  const res = await fetch(`${API_URL}/folders/${folderId}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error('Failed to delete folder')
  return res.json()
}

// ============== VLM Prompt Generation ==============

export interface GeneratedPrompts {
  success: boolean
  positive: string
  negative: string
}

// ============== Mask Preview API ==============

export interface MaskPreviewResponse {
  success: boolean
  overlay_url?: string
  error?: string
  face_detected?: boolean
  bbox?: number[]
  aspect_ratio?: number
  mask_coverage?: number
  // v7.3: Hair preview fields
  hair_region_url?: string
  hair_coverage?: number
  hair_detected?: boolean
}

export async function previewMask(
  imageId: string,
  imageType: 'face' | 'style' | 'hair',
  options?: {
    includeHairInMask?: boolean
    maskExpandPixels?: number
    maskEdgeBlur?: number
    faceMaskMethod?: FaceMaskMethod
    faceMaskBlurRadius?: number
    bboxExpandRatio?: number  // v7.2: Hair coverage ratio (1.0-3.0)
    faceImageId?: string  // v7.3: For aspect ratio adjustment (style preview)
    adjustMaskAspectRatio?: boolean  // v7.4: Enable/disable aspect ratio adjustment
  }
): Promise<MaskPreviewResponse> {
  const formData = new FormData()
  formData.append('image_id', imageId)
  formData.append('image_type', imageType)

  if (options?.includeHairInMask !== undefined) {
    formData.append('include_hair_in_mask', options.includeHairInMask.toString())
  }
  if (options?.maskExpandPixels !== undefined) {
    formData.append('mask_expand_pixels', options.maskExpandPixels.toString())
  }
  if (options?.maskEdgeBlur !== undefined) {
    formData.append('mask_edge_blur', options.maskEdgeBlur.toString())
  }
  if (options?.faceMaskMethod) {
    formData.append('face_mask_method', options.faceMaskMethod)
  }
  if (options?.faceMaskBlurRadius !== undefined) {
    formData.append('face_mask_blur_radius', options.faceMaskBlurRadius.toString())
  }
  if (options?.bboxExpandRatio !== undefined) {
    formData.append('bbox_expand_ratio', options.bboxExpandRatio.toString())
  }
  // v7.3: For aspect ratio adjustment
  if (options?.faceImageId) {
    formData.append('face_image_id', options.faceImageId)
  }
  // v7.4: Aspect ratio adjustment toggle
  if (options?.adjustMaskAspectRatio !== undefined) {
    formData.append('adjust_mask_aspect_ratio', options.adjustMaskAspectRatio.toString())
  }

  const res = await fetch(`${API_URL}/preview-mask`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || 'Failed to generate mask preview')
  }
  return res.json()
}

// ============== VLM Prompt Generation ==============

export async function generatePromptFromImages(
  faceImageId: string,
  styleImageId?: string | null
): Promise<GeneratedPrompts> {
  const formData = new FormData()
  formData.append('face_image_id', faceImageId)
  if (styleImageId) {
    formData.append('style_image_id', styleImageId)
  }

  const res = await fetch(`${API_URL}/generate-prompt`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || 'Failed to generate prompt')
  }
  return res.json()
}

// ============== PDF Report Generation ==============

export interface ReportParams {
  faceImageId: string
  styleImageId?: string | null
  resultImageUrl: string
  modelName: string
  ips: number
  loraScale: number
  styleStrength: number
  denoisingStrength: number
  inferenceSteps: number
  seed: number
  useTinyVae: boolean
  positivePrompt: string
  negativePrompt: string
  maskStyleFace: boolean
  includeHairInMask: boolean
  faceMaskMethod: string
  faceMaskBlurRadius: number
  maskExpandPixels: number
  maskEdgeBlur: number
  controlnetScale: number
  depthBlurRadius: number
  styleStrengthCap: number
  denoisingMin: number
  adjustMaskAspectRatio: boolean
}

export interface ReportResponse {
  success: boolean
  report_url: string
  filename: string
}

export async function generateReport(params: ReportParams): Promise<ReportResponse> {
  const formData = new FormData()
  formData.append('face_image_id', params.faceImageId)
  if (params.styleImageId) {
    formData.append('style_image_id', params.styleImageId)
  }
  formData.append('result_image_url', params.resultImageUrl)
  formData.append('model_name', params.modelName)
  formData.append('ips', params.ips.toString())
  formData.append('lora_scale', params.loraScale.toString())
  formData.append('style_strength', params.styleStrength.toString())
  formData.append('denoising_strength', params.denoisingStrength.toString())
  formData.append('inference_steps', params.inferenceSteps.toString())
  formData.append('seed', params.seed.toString())
  formData.append('use_tiny_vae', params.useTinyVae.toString())
  formData.append('positive_prompt', params.positivePrompt)
  formData.append('negative_prompt', params.negativePrompt)
  formData.append('mask_style_face', params.maskStyleFace.toString())
  formData.append('include_hair_in_mask', params.includeHairInMask.toString())
  formData.append('face_mask_method', params.faceMaskMethod)
  formData.append('face_mask_blur_radius', params.faceMaskBlurRadius.toString())
  formData.append('mask_expand_pixels', params.maskExpandPixels.toString())
  formData.append('mask_edge_blur', params.maskEdgeBlur.toString())
  formData.append('controlnet_scale', params.controlnetScale.toString())
  formData.append('depth_blur_radius', params.depthBlurRadius.toString())
  formData.append('style_strength_cap', params.styleStrengthCap.toString())
  formData.append('denoising_min', params.denoisingMin.toString())
  formData.append('adjust_mask_aspect_ratio', params.adjustMaskAspectRatio.toString())

  const res = await fetch(`${API_URL}/generate-report`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Unknown error' }))
    throw new Error(error.detail || 'Failed to generate report')
  }
  return res.json()
}
