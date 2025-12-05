const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8007'

export interface Model {
  id: string
  name: string
  description: string
  default?: boolean
}

export interface UploadResponse {
  success: boolean
  file_id: string
  filename: string
  url: string
}

export interface GenerateParams {
  image_id: string
  prompt: string
  model_name: string
  ips: number
  lora_scale: number
  seed: number
  style_image_id?: string | null
  style_strength?: number
}

export interface GenerateResponse {
  success: boolean
  image_url?: string
  error?: string
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
  ips: number
  lora_scale: number
  seed: number
}

export interface HistoryItem {
  id: string
  title: string
  created_at: string
  folder_id: string | null
  input_image_url: string
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

  const res = await fetch(`${API_URL}/generate`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) throw new Error('Failed to generate image')
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
