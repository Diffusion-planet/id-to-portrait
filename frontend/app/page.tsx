'use client'

import { useState, useRef, useEffect, useCallback } from 'react'
import {
  UploadIcon,
  ImageIcon,
  SparkleIcon,
  LoaderIcon,
  DownloadIcon,
  RefreshIcon,
  ChevronDownIcon,
} from '@/components/Icons'
import Sidebar from '@/components/Sidebar'
import {
  uploadImage,
  generateImage,
  getModels,
  getImageUrl,
  checkHealth,
  getHistory,
  updateHistory as updateHistoryApi,
  deleteHistory as deleteHistoryApi,
  getFolders,
  createFolder as createFolderApi,
  updateFolder as updateFolderApi,
  deleteFolder as deleteFolderApi,
  getTaskStatus,
  getActiveTasks,
  generatePromptFromImages,
  type Model,
  type HistoryItem,
  type Folder,
  type Task,
} from '@/lib/api'

// Tips for loading state
const TIPS = [
  { title: 'ID 유사도 조절', desc: 'IPS 값이 높을수록 원본 얼굴과 유사하게, 낮을수록 프롬프트에 더 충실하게 생성됩니다.' },
  { title: '프롬프트 작성 팁', desc: '영어로 작성하면 더 정확한 결과를 얻을 수 있습니다. 스타일, 배경, 조명 등을 구체적으로 묘사해보세요.' },
  { title: '시드(Seed) 활용', desc: '같은 시드 값을 사용하면 동일한 결과를 재현할 수 있습니다. 마음에 드는 결과의 시드를 기억해두세요.' },
  { title: 'LoRA 강도', desc: 'LoRA 강도가 높을수록 인물 중심의 이미지가 생성됩니다.' },
  { title: '모델 선택', desc: 'Hyper-SD는 빠른 속도와 높은 품질의 균형을 제공합니다. Turbo는 더 빠르지만 512px 해상도입니다.' },
  { title: '스타일 참조', desc: '스타일 이미지를 추가하면 배경과 분위기를 참조하여 더 자연스러운 결과물을 얻을 수 있습니다.' },
]

// Circular Dial Component - uses horizontal drag for intuitive control
function CircularDial({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  label,
  description,
}: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  label: string
  description?: string
}) {
  const [isDragging, setIsDragging] = useState(false)
  const dialRef = useRef<HTMLDivElement>(null)
  const startXRef = useRef(0)
  const startValueRef = useRef(value)

  const percentage = ((value - min) / (max - min)) * 100
  const circumference = 2 * Math.PI * 40
  const strokeDashoffset = circumference - (percentage / 100) * circumference

  const handleDragStart = useCallback((clientX: number) => {
    startXRef.current = clientX
    startValueRef.current = value
  }, [value])

  const handleDragMove = useCallback((clientX: number) => {
    const deltaX = clientX - startXRef.current
    const sensitivity = 0.005 // Adjust sensitivity
    const deltaValue = deltaX * (max - min) * sensitivity
    const newValue = startValueRef.current + deltaValue
    const steppedValue = Math.round(newValue / step) * step
    onChange(Math.max(min, Math.min(max, steppedValue)))
  }, [min, max, step, onChange])

  useEffect(() => {
    if (!isDragging) return
    const handleMove = (e: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
      handleDragMove(clientX)
    }
    const handleUp = () => setIsDragging(false)
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
    window.addEventListener('touchmove', handleMove)
    window.addEventListener('touchend', handleUp)
    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
      window.removeEventListener('touchmove', handleMove)
      window.removeEventListener('touchend', handleUp)
    }
  }, [isDragging, handleDragMove])

  return (
    <div className="card-interactive p-5 relative z-0">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="text-sm font-medium">{label}</h4>
          {description && <p className="text-xs text-neutral-400 mt-0.5">{description}</p>}
        </div>
        <span className="text-lg font-semibold tabular-nums">{value.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-6">
        <div
          ref={dialRef}
          className={`relative w-24 h-24 cursor-ew-resize select-none transition-transform ${isDragging ? 'scale-105' : ''}`}
          onMouseDown={(e) => {
            setIsDragging(true)
            handleDragStart(e.clientX)
          }}
          onTouchStart={(e) => {
            setIsDragging(true)
            handleDragStart(e.touches[0].clientX)
          }}
        >
          <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="40" className="dial-track" />
            <circle
              cx="50"
              cy="50"
              r="40"
              className="dial-progress"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
            />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <div className={`w-10 h-10 rounded-full bg-black flex items-center justify-center transition-all ${isDragging ? 'scale-110 shadow-lg' : 'shadow-md'}`}>
              <span className="text-white text-[11px] font-semibold tabular-nums">{value.toFixed(2)}</span>
            </div>
          </div>
        </div>
        <div className="flex-1 space-y-2">
          <div className="flex justify-between text-xs text-neutral-400">
            <span>{min}</span>
            <span>{max}</span>
          </div>
          <div className="value-bar-track">
            <div className="value-bar-fill" style={{ width: `${percentage}%` }} />
          </div>
          <div className="flex gap-1">
            {[0.25, 0.5, 0.75, 1].map((preset) => {
              const presetValue = Math.round((preset * max) / step) * step
              const isActive = Math.abs(value - presetValue) < 0.001
              return (
                <button
                  key={preset}
                  onClick={() => onChange(presetValue)}
                  className={`flex-1 py-1.5 text-xs rounded-md transition-all ${
                    isActive
                      ? 'bg-black text-white'
                      : 'bg-neutral-100 hover:bg-neutral-200 text-neutral-600'
                  }`}
                >
                  {presetValue.toFixed(2)}
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </div>
  )
}

// Visual Slider Component
function VisualSlider({
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  label,
  description,
}: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  step?: number
  label: string
  description?: string
}) {
  const [isHovered, setIsHovered] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const sliderRef = useRef<HTMLDivElement>(null)
  const percentage = ((value - min) / (max - min)) * 100

  const handleInteraction = useCallback((clientX: number) => {
    if (!sliderRef.current) return
    const rect = sliderRef.current.getBoundingClientRect()
    const x = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const newValue = min + x * (max - min)
    const steppedValue = Math.round(newValue / step) * step
    onChange(Math.max(min, Math.min(max, steppedValue)))
  }, [min, max, step, onChange])

  useEffect(() => {
    if (!isDragging) return
    const handleMove = (e: MouseEvent | TouchEvent) => {
      const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX
      handleInteraction(clientX)
    }
    const handleUp = () => setIsDragging(false)
    window.addEventListener('mousemove', handleMove)
    window.addEventListener('mouseup', handleUp)
    window.addEventListener('touchmove', handleMove)
    window.addEventListener('touchend', handleUp)
    return () => {
      window.removeEventListener('mousemove', handleMove)
      window.removeEventListener('mouseup', handleUp)
      window.removeEventListener('touchmove', handleMove)
      window.removeEventListener('touchend', handleUp)
    }
  }, [isDragging, handleInteraction])

  return (
    <div
      className="card-interactive p-5 relative z-0"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="text-sm font-medium">{label}</h4>
          {description && <p className="text-xs text-neutral-400 mt-0.5">{description}</p>}
        </div>
        <span className={`text-lg font-semibold tabular-nums transition-all ${isDragging ? 'scale-110' : ''}`}>
          {value.toFixed(2)}
        </span>
      </div>
      <div
        ref={sliderRef}
        className="relative h-12 cursor-pointer select-none"
        onMouseDown={(e) => {
          setIsDragging(true)
          handleInteraction(e.clientX)
        }}
        onTouchStart={(e) => {
          setIsDragging(true)
          handleInteraction(e.touches[0].clientX)
        }}
      >
        {/* Track background with segments */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 h-2 rounded-full bg-neutral-100 overflow-hidden">
          {/* Animated fill */}
          <div
            className="absolute inset-y-0 left-0 bg-black rounded-full transition-all duration-100"
            style={{ width: `${percentage}%` }}
          />
        </div>
        {/* Tick marks */}
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex justify-between px-0.5">
          {[0, 25, 50, 75, 100].map((tick) => (
            <div
              key={tick}
              className={`w-0.5 h-3 rounded-full transition-colors ${
                percentage >= tick ? 'bg-white' : 'bg-neutral-300'
              }`}
            />
          ))}
        </div>
        {/* Thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 transition-all"
          style={{ left: `${percentage}%` }}
        >
          <div className={`w-6 h-6 rounded-full bg-black shadow-lg transition-all ${
            isDragging ? 'scale-125 shadow-xl' : isHovered ? 'scale-110' : ''
          }`}>
            <div className="absolute inset-1 rounded-full bg-white" />
          </div>
        </div>
        {/* Value tooltip */}
        {(isHovered || isDragging) && (
          <div
            className="absolute -top-8 -translate-x-1/2 px-2 py-1 bg-black text-white text-xs rounded-md transition-opacity"
            style={{ left: `${percentage}%` }}
          >
            {value.toFixed(2)}
          </div>
        )}
      </div>
      {/* Labels */}
      <div className="flex justify-between mt-2 text-xs text-neutral-400">
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  )
}

// Generate default title with current date/time
function getDefaultTitle() {
  const now = new Date()
  return now.toLocaleString('ko-KR', {
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

// localStorage key for form state
const FORM_STATE_KEY = 'prometheus_form_state'

// Default prompts
const DEFAULT_PROMPT = 'professional portrait photo, natural skin texture, soft studio lighting, sharp focus, neutral background, high resolution, photorealistic'
const DEFAULT_NEGATIVE_PROMPT = 'plastic skin, waxy, smooth skin, blurry, noise, artifacts, distorted, deformed, ugly, low quality, oversaturated, cartoon, anime, painting, illustration, 3d render'

export default function Home() {
  // State - use defaults initially, load from localStorage in useEffect
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState('hyper')
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT)
  const [negativePrompt, setNegativePrompt] = useState(DEFAULT_NEGATIVE_PROMPT)
  const [isGeneratingPrompt, setIsGeneratingPrompt] = useState(false)
  const [promptTypingTarget, setPromptTypingTarget] = useState<{ positive: string; negative: string } | null>(null)
  const [ips, setIps] = useState(0.8)
  const [loraScale, setLoraScale] = useState(0.6)
  const [seed, setSeed] = useState(42)
  const [styleStrength, setStyleStrength] = useState(0.3)
  const [inferenceSteps, setInferenceSteps] = useState(4)
  const [dualAdapterMode, setDualAdapterMode] = useState(true)
  const [useTinyVae, setUseTinyVae] = useState(false) // Full VAE by default for better quality
  const [title, setTitle] = useState('')
  const [isHydrated, setIsHydrated] = useState(false)

  const [uploadedImage, setUploadedImage] = useState<{ id: string; url: string } | null>(null)
  const [styleImage, setStyleImage] = useState<{ id: string; url: string } | null>(null)
  const [generatedImage, setGeneratedImage] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [isGenerating, setIsGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking')
  const [showModelDropdown, setShowModelDropdown] = useState(false)
  const [isDragOver, setIsDragOver] = useState(false)
  const [isStyleDragOver, setIsStyleDragOver] = useState(false)
  const [isUploadingStyle, setIsUploadingStyle] = useState(false)
  const [currentTip, setCurrentTip] = useState(0)

  // Task tracking for generation
  const [activeTask, setActiveTask] = useState<Task | null>(null)
  const pollingRef = useRef<NodeJS.Timeout | null>(null)
  const [displayedMessage, setDisplayedMessage] = useState('')
  const [targetMessage, setTargetMessage] = useState('')

  // History and Sidebar state
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [folders, setFolders] = useState<Folder[]>([])
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)
  const [viewingHistory, setViewingHistory] = useState<HistoryItem | null>(null)

  const fileInputRef = useRef<HTMLInputElement>(null)
  const styleFileInputRef = useRef<HTMLInputElement>(null)
  const dropdownRef = useRef<HTMLDivElement>(null)

  // Poll for task status
  const startPolling = useCallback((taskId: string) => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current)
    }

    const poll = async () => {
      try {
        const task = await getTaskStatus(taskId)
        setActiveTask(task)

        if (task.status === 'completed') {
          setIsGenerating(false)
          setGeneratedImage(task.image_url ? getImageUrl(task.image_url) : null)
          // Refresh history to get the new item
          const historyRes = await getHistory()
          setHistory(historyRes.history)
          if (task.history_id) {
            setSelectedHistoryId(task.history_id)
          }
          setActiveTask(null)
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
        } else if (task.status === 'failed') {
          setIsGenerating(false)
          setError(task.error || 'Generation failed')
          setActiveTask(null)
          if (pollingRef.current) {
            clearInterval(pollingRef.current)
            pollingRef.current = null
          }
        }
      } catch (e) {
        console.error('Polling error:', e)
      }
    }

    poll() // Initial poll
    pollingRef.current = setInterval(poll, 1000)
  }, [])

  // Check API health and fetch models, history, folders
  useEffect(() => {
    const init = async () => {
      try {
        await checkHealth()
        setApiStatus('online')
        const [modelsRes, historyRes, foldersRes, tasksRes] = await Promise.all([
          getModels(),
          getHistory(),
          getFolders(),
          getActiveTasks(),
        ])
        setModels(modelsRes.models)
        setHistory(historyRes.history)
        setFolders(foldersRes.folders)

        // Resume any active task
        if (tasksRes.tasks.length > 0) {
          const task = tasksRes.tasks[0]
          setActiveTask(task)
          setIsGenerating(true)
          // Restore input images from task
          if (task.input_image_url) {
            const imageId = task.params.image_id
            setUploadedImage({ id: imageId, url: task.input_image_url })
          }
          if (task.style_image_url && task.params.style_image_id) {
            setStyleImage({ id: task.params.style_image_id, url: task.style_image_url })
          }
          startPolling(task.id)
        }
      } catch {
        setApiStatus('offline')
      }
    }
    init()

    // Cleanup polling on unmount
    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current)
      }
    }
  }, [startPolling])

  // Tip carousel during generation
  useEffect(() => {
    if (!isGenerating) return
    const interval = setInterval(() => {
      setCurrentTip((prev) => (prev + 1) % TIPS.length)
    }, 4000)
    return () => clearInterval(interval)
  }, [isGenerating])

  // Update target message when activeTask changes
  useEffect(() => {
    if (activeTask?.progress_message) {
      setTargetMessage(activeTask.progress_message)
    }
  }, [activeTask?.progress_message])

  // Typing effect for progress message
  useEffect(() => {
    if (!targetMessage) {
      setDisplayedMessage('')
      return
    }

    // If displayed message is a prefix of target, continue typing
    if (targetMessage.startsWith(displayedMessage) && displayedMessage !== targetMessage) {
      const timeout = setTimeout(() => {
        setDisplayedMessage(targetMessage.slice(0, displayedMessage.length + 1))
      }, 30)
      return () => clearTimeout(timeout)
    }

    // If target changed completely, start fresh
    if (!targetMessage.startsWith(displayedMessage)) {
      setDisplayedMessage('')
    }
  }, [targetMessage, displayedMessage])

  // Typing effect for AI-generated prompts
  useEffect(() => {
    if (!promptTypingTarget) return

    const { positive, negative } = promptTypingTarget

    // Type positive prompt first, then negative
    if (prompt.length < positive.length) {
      const timeout = setTimeout(() => {
        setPrompt(positive.slice(0, prompt.length + 1))
      }, 15)
      return () => clearTimeout(timeout)
    } else if (negativePrompt.length < negative.length) {
      const timeout = setTimeout(() => {
        setNegativePrompt(negative.slice(0, negativePrompt.length + 1))
      }, 15)
      return () => clearTimeout(timeout)
    } else {
      // Done typing
      setPromptTypingTarget(null)
      setIsGeneratingPrompt(false)
    }
  }, [promptTypingTarget, prompt, negativePrompt])

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setShowModelDropdown(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  // Load form state from localStorage after hydration
  useEffect(() => {
    try {
      const saved = localStorage.getItem(FORM_STATE_KEY)
      if (saved) {
        const state = JSON.parse(saved)
        if (state.selectedModel) setSelectedModel(state.selectedModel)
        if (state.prompt) setPrompt(state.prompt)
        if (state.negativePrompt) setNegativePrompt(state.negativePrompt)
        if (state.ips !== undefined) setIps(state.ips)
        if (state.loraScale !== undefined) setLoraScale(state.loraScale)
        if (state.seed !== undefined) setSeed(state.seed)
        if (state.styleStrength !== undefined) setStyleStrength(state.styleStrength)
        if (state.inferenceSteps !== undefined) setInferenceSteps(state.inferenceSteps)
        if (state.dualAdapterMode !== undefined) setDualAdapterMode(state.dualAdapterMode)
        if (state.useTinyVae !== undefined) setUseTinyVae(state.useTinyVae)
      }
    } catch {
      // Ignore parse errors
    }
    setIsHydrated(true)
  }, [])

  // Save form state to localStorage (only after hydration)
  useEffect(() => {
    if (!isHydrated) return
    const state = {
      selectedModel,
      prompt,
      negativePrompt,
      ips,
      loraScale,
      seed,
      styleStrength,
      inferenceSteps,
      dualAdapterMode,
      useTinyVae,
    }
    try {
      localStorage.setItem(FORM_STATE_KEY, JSON.stringify(state))
    } catch {
      // Ignore storage errors (quota exceeded, etc.)
    }
  }, [isHydrated, selectedModel, prompt, negativePrompt, ips, loraScale, seed, styleStrength, inferenceSteps, dualAdapterMode, useTinyVae])

  // Handle file upload for face reference
  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드할 수 있습니다.')
      return
    }

    setIsUploading(true)
    setError(null)

    try {
      const result = await uploadImage(file)
      setUploadedImage({
        id: result.file_id,
        url: getImageUrl(result.url),
      })
    } catch {
      setError('이미지 업로드에 실패했습니다.')
    } finally {
      setIsUploading(false)
    }
  }, [])

  // Handle file upload for style reference
  const handleStyleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError('이미지 파일만 업로드할 수 있습니다.')
      return
    }

    setIsUploadingStyle(true)
    setError(null)

    try {
      const result = await uploadImage(file)
      setStyleImage({
        id: result.file_id,
        url: getImageUrl(result.url),
      })
    } catch {
      setError('스타일 이미지 업로드에 실패했습니다.')
    } finally {
      setIsUploadingStyle(false)
    }
  }, [])

  // Handle drag and drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  // Handle drag and drop for style image
  const handleStyleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsStyleDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleStyleFileSelect(file)
  }, [handleStyleFileSelect])

  const handleStyleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsStyleDragOver(true)
  }, [])

  const handleStyleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsStyleDragOver(false)
  }, [])

  // Generate image
  const handleGenerate = async () => {
    if (!uploadedImage || !prompt.trim()) {
      setError('이미지와 프롬프트를 모두 입력해주세요.')
      return
    }

    setIsGenerating(true)
    setError(null)
    setGeneratedImage(null)
    setCurrentTip(0)
    setSelectedHistoryId(null)
    setViewingHistory(null)
    setDisplayedMessage('')
    setTargetMessage('')

    try {
      const result = await generateImage({
        image_id: uploadedImage.id,
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim() || undefined,
        model_name: selectedModel,
        ips,
        lora_scale: loraScale,
        seed,
        style_image_id: styleImage?.id || null,
        style_strength: styleImage ? styleStrength : undefined,
        inference_steps: inferenceSteps,
        dual_adapter_mode: styleImage ? dualAdapterMode : false,
        title: title.trim() || getDefaultTitle(),
        use_tiny_vae: useTinyVae,
      })

      if (result.success && result.task_id) {
        // Start polling for task completion
        startPolling(result.task_id)
        setTitle('')
      } else {
        setError('이미지 생성 요청에 실패했습니다.')
        setIsGenerating(false)
      }
    } catch {
      setError('이미지 생성 중 오류가 발생했습니다.')
      setIsGenerating(false)
    }
  }

  // Return to active generation from history view
  const handleReturnToGeneration = () => {
    setViewingHistory(null)
    setSelectedHistoryId(activeTask?.history_id || null)

    // Restore input images from active task
    if (activeTask) {
      if (activeTask.input_image_url) {
        setUploadedImage({ id: activeTask.params.image_id, url: activeTask.input_image_url })
      }
      if (activeTask.style_image_url && activeTask.params.style_image_id) {
        setStyleImage({ id: activeTask.params.style_image_id, url: activeTask.style_image_url })
      }
      // Clear generated image since we're still generating
      setGeneratedImage(null)
    }
  }

  // Download generated image
  const handleDownload = async () => {
    if (!generatedImage) return

    const response = await fetch(generatedImage)
    const blob = await response.blob()
    const url = window.URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `prometheus_${Date.now()}.png`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(url)
  }

  // Randomize seed
  const randomizeSeed = () => {
    setSeed(Math.floor(Math.random() * 999999))
  }

  // Generate prompts using VLM (Gemini)
  const handleGeneratePrompt = async () => {
    if (!uploadedImage) {
      setError('프롬프트 생성을 위해 먼저 얼굴 이미지를 업로드해주세요.')
      return
    }

    setIsGeneratingPrompt(true)
    setError(null)
    // Clear existing prompts for typing effect
    setPrompt('')
    setNegativePrompt('')

    try {
      const result = await generatePromptFromImages(
        uploadedImage.id,
        styleImage?.id || null
      )

      if (result.success) {
        // Start typing effect
        setPromptTypingTarget({
          positive: result.positive,
          negative: result.negative,
        })
      } else {
        setError('프롬프트 생성에 실패했습니다.')
        setIsGeneratingPrompt(false)
      }
    } catch (e) {
      console.error('Prompt generation error:', e)
      setError('프롬프트 생성 중 오류가 발생했습니다.')
      setIsGeneratingPrompt(false)
    }
  }

  // Start new generation (clear current state)
  const handleNewGeneration = () => {
    setUploadedImage(null)
    setStyleImage(null)
    setGeneratedImage(null)
    setSelectedHistoryId(null)
    setViewingHistory(null)
    setTitle('')
    setError(null)
    // Keep other settings (prompt, model, etc.) as they are
  }

  // History handlers
  const handleSelectHistory = (item: HistoryItem) => {
    setSelectedHistoryId(item.id)
    setViewingHistory(item)

    // Extract file ID from input image URL (e.g., /uploads/uuid.ext -> uuid)
    const inputPath = item.input_image_url
    const fileName = inputPath.split('/').pop() || ''
    const fileId = fileName.split('.')[0] || ''
    setUploadedImage({ id: fileId, url: getImageUrl(item.input_image_url) })

    // Restore style image if exists
    if (item.style_image_url) {
      const stylePath = item.style_image_url
      const styleFileName = stylePath.split('/').pop() || ''
      const styleFileId = styleFileName.split('.')[0] || ''
      setStyleImage({ id: styleFileId, url: getImageUrl(item.style_image_url) })
    } else {
      setStyleImage(null)
    }

    setGeneratedImage(getImageUrl(item.output_image_url))
    setPrompt(item.settings.prompt)
    setNegativePrompt(item.settings.negative_prompt || '')
    setSelectedModel(item.settings.model_name)
    setIps(item.settings.ips)
    setLoraScale(item.settings.lora_scale)
    setSeed(item.settings.seed)
    setStyleStrength(item.settings.style_strength ?? 0.3)
    setInferenceSteps(item.settings.inference_steps ?? 4)
    setDualAdapterMode(item.settings.dual_adapter_mode ?? true)
    setUseTinyVae(item.settings.use_tiny_vae ?? false)
    setTitle(item.title)
    setSidebarOpen(false)
  }

  const handleDeleteHistory = async (id: string) => {
    try {
      await deleteHistoryApi(id)
      setHistory((prev) => prev.filter((h) => h.id !== id))
      if (selectedHistoryId === id) {
        setSelectedHistoryId(null)
      }
    } catch (e) {
      console.error('Failed to delete history:', e)
    }
  }

  const handleUpdateHistoryTitle = async (id: string, newTitle: string) => {
    try {
      const { item } = await updateHistoryApi(id, { title: newTitle })
      setHistory((prev) => prev.map((h) => (h.id === id ? item : h)))
    } catch (e) {
      console.error('Failed to update history:', e)
    }
  }

  const handleMoveToFolder = async (historyId: string, folderId: string | null) => {
    try {
      const { item } = await updateHistoryApi(historyId, { folder_id: folderId })
      setHistory((prev) => prev.map((h) => (h.id === historyId ? item : h)))
    } catch (e) {
      console.error('Failed to move history:', e)
    }
  }

  // Folder handlers
  const handleCreateFolder = async (name: string) => {
    try {
      const { folder } = await createFolderApi(name)
      setFolders((prev) => [...prev, folder])
    } catch (e) {
      console.error('Failed to create folder:', e)
    }
  }

  const handleDeleteFolder = async (id: string) => {
    try {
      await deleteFolderApi(id)
      setFolders((prev) => prev.filter((f) => f.id !== id))
      // Move items from deleted folder to unfiled
      setHistory((prev) =>
        prev.map((h) => (h.folder_id === id ? { ...h, folder_id: null } : h))
      )
    } catch (e) {
      console.error('Failed to delete folder:', e)
    }
  }

  const handleRenameFolder = async (id: string, name: string) => {
    try {
      const { folder } = await updateFolderApi(id, { name })
      setFolders((prev) => prev.map((f) => (f.id === id ? folder : f)))
    } catch (e) {
      console.error('Failed to rename folder:', e)
    }
  }

  const selectedModelData = models.find(m => m.id === selectedModel)

  return (
    <div className="min-h-screen">
      {/* Sidebar */}
      <Sidebar
        isOpen={sidebarOpen}
        onToggle={() => setSidebarOpen(!sidebarOpen)}
        history={history}
        folders={folders}
        selectedHistoryId={selectedHistoryId}
        onSelectHistory={handleSelectHistory}
        onDeleteHistory={handleDeleteHistory}
        onUpdateHistoryTitle={handleUpdateHistoryTitle}
        onMoveToFolder={handleMoveToFolder}
        onCreateFolder={handleCreateFolder}
        onDeleteFolder={handleDeleteFolder}
        onRenameFolder={handleRenameFolder}
      />

      {/* Header */}
      <header className="glass sticky top-0 z-30 border-b border-neutral-200/50">
        <div className="px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2.5">
              <img
                src="/logo.svg"
                alt="Prometheus"
                className="h-8 w-auto"
              />
              <h1 className="text-base font-bold font-logo">Prometheus</h1>
            </div>
            <span className="text-[11px] text-neutral-500 px-2 py-1 bg-white/80 rounded-md font-medium shadow-sm">
              FastFace
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full transition-all ${
              apiStatus === 'online' ? 'bg-emerald-50' : apiStatus === 'offline' ? 'bg-red-50' : 'bg-amber-50'
            }`}>
              <span
                className={`w-2 h-2 rounded-full ${
                  apiStatus === 'online'
                    ? 'bg-emerald-500'
                    : apiStatus === 'offline'
                    ? 'bg-red-500'
                    : 'bg-amber-500 animate-pulse'
                }`}
              />
              <span className="text-xs font-medium text-neutral-600">
                {apiStatus === 'online' ? 'Connected' : apiStatus === 'offline' ? 'Offline' : 'Connecting...'}
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="p-6 max-w-7xl mx-auto">
        {/* Return to generation banner */}
        {isGenerating && viewingHistory && (
          <div className="mb-4 p-4 bg-black text-white rounded-2xl flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-white/20 flex items-center justify-center">
                <LoaderIcon size={16} className="text-white" />
              </div>
              <div>
                <p className="text-sm font-medium">이미지 생성 중...</p>
                <p className="text-xs text-white/60">히스토리를 보는 중입니다</p>
              </div>
            </div>
            <button
              onClick={handleReturnToGeneration}
              className="px-4 py-2 bg-white text-black text-sm font-medium rounded-xl hover:bg-neutral-100 transition-colors"
            >
              생성 화면으로 돌아가기
            </button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Input */}
          <div className="space-y-6">
            {/* Title Input */}
            <section className="card-elevated p-4">
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder={getDefaultTitle()}
                  className="flex-1 text-lg font-medium bg-transparent border-0 focus:outline-none placeholder:text-neutral-300"
                />
                {(uploadedImage || generatedImage) && (
                  <button
                    onClick={handleNewGeneration}
                    className="flex items-center gap-1.5 px-3.5 py-2 text-sm font-medium bg-neutral-100 hover:bg-black hover:text-white text-neutral-700 rounded-xl transition-all whitespace-nowrap shadow-sm hover:shadow-md btn-press"
                  >
                    <RefreshIcon size={14} />
                    <span>새로 시작</span>
                  </button>
                )}
              </div>
              <p className="text-xs text-neutral-400 mt-1">비워두면 현재 시간으로 저장됩니다</p>
            </section>

            {/* Dual Image Upload */}
            <section className="card-elevated p-6">
              <div className="mb-5">
                <h3 className="text-sm font-semibold">참조 이미지</h3>
                <p className="text-xs text-neutral-400 mt-0.5">얼굴과 스타일을 각각 참조할 이미지를 업로드하세요</p>
              </div>

              <div className="relative flex items-stretch gap-3">
                {/* Face Reference */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-neutral-600">얼굴 참조</span>
                    {uploadedImage && (
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="text-[10px] text-neutral-400 hover:text-black transition-colors"
                      >
                        변경
                      </button>
                    )}
                  </div>
                  <div
                    className={`relative rounded-2xl overflow-hidden transition-all duration-300 ${
                      uploadedImage
                        ? 'bg-neutral-100'
                        : isDragOver
                        ? 'bg-black/5 ring-2 ring-black scale-[1.02]'
                        : 'bg-neutral-50 hover:bg-neutral-100'
                    }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                  >
                    {uploadedImage ? (
                      <div className="relative group aspect-[3/4]">
                        <img
                          src={uploadedImage.url}
                          alt="Face reference"
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-all duration-300" />
                      </div>
                    ) : (
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="w-full py-16 flex flex-col items-center justify-center gap-3 transition-all btn-press"
                      >
                        {isUploading ? (
                          <div className="flex flex-col items-center gap-2">
                            <div className="w-10 h-10 rounded-xl bg-black flex items-center justify-center shadow-lg animate-pulse">
                              <LoaderIcon size={16} className="text-white" />
                            </div>
                            <span className="text-xs text-neutral-400">업로드 중...</span>
                          </div>
                        ) : (
                          <>
                            <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all shadow-md ${
                              isDragOver ? 'bg-black scale-110' : 'bg-black/90'
                            }`}>
                              <UploadIcon size={18} className="text-white" />
                            </div>
                            <div className="text-center px-2">
                              <p className="text-xs font-medium text-neutral-600">
                                {isDragOver ? '놓으세요' : '얼굴 이미지'}
                              </p>
                              <p className="text-[10px] text-neutral-400 mt-0.5">인물의 얼굴</p>
                            </div>
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>

                {/* Center Merge Indicator with Circular Flow */}
                <div className="flex items-center justify-center w-24 shrink-0 relative">
                  {/* SVG with elegant circular flow */}
                  <svg viewBox="0 0 100 100" className="absolute inset-0 w-full h-full" style={{ overflow: 'visible' }}>
                    {/* Top arc - flows left to right */}
                    <path
                      d="M 20 50 A 30 30 0 0 1 80 50"
                      fill="none"
                      stroke="#e5e5e5"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      className={`transition-opacity duration-500 ${uploadedImage && styleImage ? 'opacity-100' : 'opacity-0'}`}
                    />
                    {/* Top arc flowing particle */}
                    <path
                      d="M 20 50 A 30 30 0 0 1 80 50"
                      fill="none"
                      stroke="#000"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeDasharray="12 82"
                      className={`transition-opacity duration-500 ${uploadedImage && styleImage ? 'opacity-70 animate-flow-arc-lr' : 'opacity-0'}`}
                    />

                    {/* Bottom arc - flows right to left */}
                    <path
                      d="M 80 50 A 30 30 0 0 1 20 50"
                      fill="none"
                      stroke="#e5e5e5"
                      strokeWidth="1.5"
                      strokeLinecap="round"
                      className={`transition-opacity duration-500 ${uploadedImage && styleImage ? 'opacity-100' : 'opacity-0'}`}
                    />
                    {/* Bottom arc flowing particle */}
                    <path
                      d="M 80 50 A 30 30 0 0 1 20 50"
                      fill="none"
                      stroke="#000"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeDasharray="12 82"
                      className={`transition-opacity duration-500 ${uploadedImage && styleImage ? 'opacity-70 animate-flow-arc-rl' : 'opacity-0'}`}
                    />
                  </svg>

                  {/* Center orb with logo */}
                  <div className={`relative z-10 transition-all duration-500 ${
                    uploadedImage && styleImage ? 'opacity-100 scale-100' : 'opacity-30 scale-90'
                  }`}>
                    <div className={`w-12 h-12 rounded-full bg-black shadow-lg flex items-center justify-center overflow-hidden ${
                      uploadedImage && styleImage ? 'animate-heartbeat' : ''
                    }`}>
                      <img
                        src="/logo.svg"
                        alt="Prometheus"
                        className="w-7 h-7 object-contain"
                      />
                    </div>
                    {/* Heartbeat pulse rings */}
                    {uploadedImage && styleImage && (
                      <>
                        <div className="absolute inset-0 rounded-full bg-black/20 animate-ping" />
                        <div className="absolute -inset-2 rounded-full border border-black/10 animate-pulse" />
                      </>
                    )}
                  </div>
                </div>

                {/* Style Reference */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-medium text-neutral-600">스타일 참조</span>
                    {styleImage && (
                      <button
                        onClick={() => styleFileInputRef.current?.click()}
                        className="text-[10px] text-neutral-400 hover:text-black transition-colors"
                      >
                        변경
                      </button>
                    )}
                  </div>
                  <div
                    className={`relative rounded-2xl overflow-hidden transition-all duration-300 ${
                      styleImage
                        ? 'bg-neutral-100'
                        : isStyleDragOver
                        ? 'bg-black/5 ring-2 ring-black scale-[1.02]'
                        : 'bg-neutral-50 hover:bg-neutral-100'
                    }`}
                    onDrop={handleStyleDrop}
                    onDragOver={handleStyleDragOver}
                    onDragLeave={handleStyleDragLeave}
                  >
                    {styleImage ? (
                      <div className="relative group aspect-[3/4]">
                        <img
                          src={styleImage.url}
                          alt="Style reference"
                          className="w-full h-full object-cover"
                        />
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-all duration-300" />
                      </div>
                    ) : (
                      <button
                        onClick={() => styleFileInputRef.current?.click()}
                        disabled={isUploadingStyle}
                        className="w-full py-16 flex flex-col items-center justify-center gap-3 transition-all btn-press"
                      >
                        {isUploadingStyle ? (
                          <div className="flex flex-col items-center gap-2">
                            <div className="w-10 h-10 rounded-xl bg-black flex items-center justify-center shadow-lg animate-pulse">
                              <LoaderIcon size={16} className="text-white" />
                            </div>
                            <span className="text-xs text-neutral-400">업로드 중...</span>
                          </div>
                        ) : (
                          <>
                            <div className={`w-12 h-12 rounded-xl flex items-center justify-center transition-all shadow-md border-2 border-dashed ${
                              isStyleDragOver ? 'bg-black scale-110 border-transparent' : 'bg-white border-neutral-200'
                            }`}>
                              <ImageIcon size={18} className={isStyleDragOver ? 'text-white' : 'text-neutral-400'} />
                            </div>
                            <div className="text-center px-2">
                              <p className="text-xs font-medium text-neutral-600">
                                {isStyleDragOver ? '놓으세요' : '스타일 이미지'}
                              </p>
                              <p className="text-[10px] text-neutral-400 mt-0.5">선택사항</p>
                            </div>
                          </>
                        )}
                      </button>
                    )}
                  </div>
                </div>
              </div>

              {/* Info text */}
              <div className="mt-4 p-3 bg-neutral-50 rounded-xl">
                <p className="text-xs text-neutral-500 leading-relaxed">
                  <span className="font-medium text-neutral-700">얼굴</span>은 인물의 아이덴티티를,
                  <span className="font-medium text-neutral-700"> 스타일</span>은 배경과 분위기를 참조합니다.
                  스타일 이미지는 선택사항입니다.
                </p>
              </div>

              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) handleFileSelect(file)
                }}
              />
              <input
                ref={styleFileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0]
                  if (file) handleStyleFileSelect(file)
                }}
              />
            </section>

            {/* Prompt */}
            <section className="card-elevated p-6">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-sm font-semibold">프롬프트</h3>
                  <p className="text-xs text-neutral-400 mt-0.5">생성하고 싶은 이미지를 설명하세요</p>
                </div>
                <button
                  onClick={handleGeneratePrompt}
                  disabled={!uploadedImage || isGeneratingPrompt}
                  className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium bg-neutral-100 hover:bg-black hover:text-white text-neutral-600 rounded-lg transition-all disabled:opacity-50 disabled:cursor-not-allowed btn-press"
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
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                placeholder="A professional portrait photo with soft studio lighting, neutral gray background..."
                className="w-full h-24 px-4 py-3 text-sm bg-neutral-50 border-0 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-black/10 focus:bg-white transition-all placeholder:text-neutral-400"
              />

              {/* Negative Prompt */}
              <div className="mt-4 pt-4 border-t border-neutral-100">
                <div className="mb-3">
                  <h4 className="text-xs font-medium text-neutral-500">Negative Prompt</h4>
                  <p className="text-xs text-neutral-400 mt-0.5">피하고 싶은 요소를 입력하세요</p>
                </div>
                <textarea
                  value={negativePrompt}
                  onChange={(e) => setNegativePrompt(e.target.value)}
                  placeholder="plastic skin, blurry, artifacts, distorted..."
                  className="w-full h-20 px-4 py-3 text-sm bg-neutral-50 border-0 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-black/10 focus:bg-white transition-all placeholder:text-neutral-400"
                />
              </div>
            </section>

            {/* Model Selection */}
            <section className="card-elevated p-6 relative z-20">
              <div className="mb-4">
                <h3 className="text-sm font-semibold">모델 선택</h3>
                <p className="text-xs text-neutral-400 mt-0.5">생성 속도와 품질이 달라집니다</p>
              </div>
              <div ref={dropdownRef} className="relative">
                <button
                  onClick={() => setShowModelDropdown(!showModelDropdown)}
                  className={`w-full px-4 py-3.5 text-sm text-left bg-neutral-50 rounded-xl flex items-center justify-between transition-all btn-press ${
                    showModelDropdown ? 'ring-2 ring-black/10 bg-white' : 'hover:bg-neutral-100'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-lg bg-black flex items-center justify-center">
                      <SparkleIcon size={14} className="text-white" />
                    </div>
                    <div>
                      <span className="font-medium block">{selectedModelData?.name || 'Select model'}</span>
                      {selectedModelData && (
                        <span className="text-neutral-400 text-xs">{selectedModelData.description}</span>
                      )}
                    </div>
                  </div>
                  <ChevronDownIcon
                    size={18}
                    className={`text-neutral-400 transition-transform duration-300 ${showModelDropdown ? 'rotate-180' : ''}`}
                  />
                </button>
                {showModelDropdown && (
                  <div className="absolute z-30 w-full mt-2 bg-white rounded-2xl shadow-2xl overflow-hidden border border-neutral-100">
                    {models.map((model, idx) => (
                      <button
                        key={model.id}
                        onClick={() => {
                          setSelectedModel(model.id)
                          // Reset inference steps to model's default when changing models
                          if (model.default_steps) {
                            setInferenceSteps(model.default_steps)
                          } else if (model.valid_steps && model.valid_steps.length > 0) {
                            setInferenceSteps(model.valid_steps[Math.floor(model.valid_steps.length / 2)])
                          }
                          setShowModelDropdown(false)
                        }}
                        className={`w-full px-4 py-3.5 text-sm text-left transition-all flex items-center gap-3 ${
                          selectedModel === model.id
                            ? 'bg-black text-white'
                            : 'hover:bg-neutral-50'
                        } ${idx !== models.length - 1 ? 'border-b border-neutral-100' : ''}`}
                      >
                        <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                          selectedModel === model.id ? 'bg-white/20' : 'bg-neutral-100'
                        }`}>
                          <SparkleIcon size={14} className={selectedModel === model.id ? 'text-white' : 'text-neutral-400'} />
                        </div>
                        <div>
                          <div className="font-medium">{model.name}</div>
                          <div className={`text-xs mt-0.5 ${selectedModel === model.id ? 'text-neutral-300' : 'text-neutral-400'}`}>
                            {model.description}
                          </div>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </section>

            {/* Parameters */}
            <section className="space-y-4">
              <div className="flex items-center justify-between px-1">
                <h3 className="text-sm font-semibold">세부 설정</h3>
              </div>

              {/* ID Similarity - Circular Dial */}
              <CircularDial
                value={ips}
                onChange={setIps}
                min={0}
                max={1.5}
                step={0.05}
                label="ID 유사도"
                description="원본 얼굴과의 유사도를 조절합니다"
              />

              {/* LoRA Scale - Visual Slider */}
              <VisualSlider
                value={loraScale}
                onChange={setLoraScale}
                min={0}
                max={1}
                step={0.05}
                label="LoRA 강도"
                description="인물 중심 생성 강도를 조절합니다"
              />

              {/* Style Settings - Only show when style image is uploaded */}
              {styleImage && (
                <>
                  {/* Dual Adapter Mode Toggle */}
                  <div className="card-interactive p-5 relative z-0">
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="text-sm font-medium">Dual Adapter Mode</h4>
                        <p className="text-xs text-neutral-400 mt-0.5">
                          {dualAdapterMode
                            ? '얼굴과 스타일을 분리해서 처리합니다 (권장)'
                            : 'img2img 방식으로 스타일을 적용합니다'}
                        </p>
                      </div>
                      <button
                        onClick={() => setDualAdapterMode(!dualAdapterMode)}
                        className={`relative w-12 h-7 rounded-full transition-all ${
                          dualAdapterMode ? 'bg-black' : 'bg-neutral-200'
                        }`}
                      >
                        <span
                          className={`absolute top-1 w-5 h-5 bg-white rounded-full shadow transition-all ${
                            dualAdapterMode ? 'left-6' : 'left-1'
                          }`}
                        />
                      </button>
                    </div>
                  </div>

                  <VisualSlider
                    value={styleStrength}
                    onChange={setStyleStrength}
                    min={0}
                    max={1}
                    step={0.05}
                    label="스타일 강도"
                    description={dualAdapterMode
                      ? '스타일 CLIP 임베딩의 영향력을 조절합니다'
                      : '스타일 이미지의 영향력을 조절합니다'}
                  />
                </>
              )}

              {/* Inference Steps */}
              <div className="card-interactive p-5 relative z-0">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h4 className="text-sm font-medium">생성 스텝</h4>
                    <p className="text-xs text-neutral-400 mt-0.5">높을수록 품질 향상, 속도 감소</p>
                  </div>
                  <span className="text-sm font-mono font-semibold text-black">{inferenceSteps}</span>
                </div>
                <div className="flex gap-2">
                  {(selectedModelData?.valid_steps || [1, 2, 4, 8]).map((step) => (
                    <button
                      key={step}
                      onClick={() => setInferenceSteps(step)}
                      className={`flex-1 py-2.5 text-sm font-medium rounded-xl transition-all ${
                        inferenceSteps === step
                          ? 'bg-black text-white shadow-lg'
                          : 'bg-neutral-100 hover:bg-neutral-200 text-neutral-600'
                      }`}
                    >
                      {step}
                    </button>
                  ))}
                </div>
                <p className="text-xs text-neutral-400 mt-3 text-center">
                  {selectedModelData?.name || '모델'}에서 지원하는 스텝: {(selectedModelData?.valid_steps || []).join(', ')}
                </p>
              </div>

              {/* VAE Selection */}
              <div className="card-interactive p-5 relative z-0">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-sm font-medium">VAE 디코더</h4>
                    <p className="text-xs text-neutral-400 mt-0.5">
                      {useTinyVae
                        ? 'TinyVAE: 빠른 속도, 약간의 품질 손실'
                        : 'Full VAE: 최고 품질, 느린 속도'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-medium ${!useTinyVae ? 'text-black' : 'text-neutral-400'}`}>Full</span>
                    <button
                      onClick={() => setUseTinyVae(!useTinyVae)}
                      className={`relative w-12 h-7 rounded-full transition-all ${
                        useTinyVae ? 'bg-black' : 'bg-neutral-200'
                      }`}
                    >
                      <span
                        className={`absolute top-1 w-5 h-5 bg-white rounded-full shadow transition-all ${
                          useTinyVae ? 'left-6' : 'left-1'
                        }`}
                      />
                    </button>
                    <span className={`text-xs font-medium ${useTinyVae ? 'text-black' : 'text-neutral-400'}`}>Tiny</span>
                  </div>
                </div>
                <div className="mt-3 p-3 bg-neutral-50 rounded-xl">
                  <p className="text-xs text-neutral-500 leading-relaxed">
                    {useTinyVae
                      ? 'TinyVAE는 디코딩 속도가 빠르지만, 피부 질감과 세밀한 디테일이 약간 뭉개질 수 있습니다.'
                      : 'Full VAE는 피부 질감, 머리카락 등 세밀한 디테일을 최대한 보존합니다. 최종 결과물에 권장됩니다.'}
                  </p>
                </div>
              </div>

              {/* Seed */}
              <div className="card-interactive p-5 relative z-0">
                <div className="flex items-start justify-between mb-4">
                  <div>
                    <h4 className="text-sm font-medium">시드</h4>
                    <p className="text-xs text-neutral-400 mt-0.5">결과 재현을 위한 고유 번호</p>
                  </div>
                  <button
                    onClick={randomizeSeed}
                    className="flex items-center gap-2 px-3 py-1.5 bg-black text-white text-xs font-medium rounded-lg hover:bg-neutral-800 transition-all btn-press"
                  >
                    <RefreshIcon size={12} />
                    <span>랜덤</span>
                  </button>
                </div>
                <input
                  type="number"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
                  className="w-full px-4 py-3 text-sm font-mono bg-neutral-50 border-0 rounded-xl focus:outline-none focus:ring-2 focus:ring-black/10 focus:bg-white transition-all"
                />
              </div>
            </section>

            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !uploadedImage || !prompt.trim() || apiStatus !== 'online'}
              className="w-full py-4 bg-black text-white text-sm font-semibold rounded-2xl flex items-center justify-center gap-2.5 hover:bg-neutral-900 active:scale-[0.98] disabled:bg-neutral-200 disabled:text-neutral-400 disabled:cursor-not-allowed transition-all shadow-lg shadow-black/20 btn-press glow-on-hover"
            >
              {isGenerating ? (
                <>
                  <LoaderIcon size={18} />
                  <span>생성 중...</span>
                </>
              ) : (
                <>
                  <SparkleIcon size={18} />
                  <span>이미지 생성</span>
                </>
              )}
            </button>

            {/* Error Message */}
            {error && (
              <div className="p-4 bg-red-50 border border-red-100 rounded-xl">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Output */}
          <div className="lg:sticky lg:top-24 lg:self-start">
            <div className="card-elevated overflow-hidden">
              <div className="p-6 border-b border-neutral-100">
                <h3 className="text-sm font-semibold">생성 결과</h3>
                <p className="text-xs text-neutral-400 mt-0.5">생성된 이미지가 여기에 표시됩니다</p>
              </div>
              <div className="relative min-h-[520px] bg-neutral-50">
                {generatedImage ? (
                  <div className="relative group">
                    <img
                      src={generatedImage}
                      alt="Generated"
                      className="w-full h-auto"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300">
                      <div className="absolute bottom-0 left-0 right-0 p-5">
                        <button
                          onClick={handleDownload}
                          className="flex items-center gap-2.5 px-5 py-2.5 bg-white hover:bg-neutral-100 text-sm font-medium rounded-xl transition-all shadow-lg btn-press"
                        >
                          <DownloadIcon size={16} />
                          <span>다운로드</span>
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center p-8">
                    {isGenerating ? (
                      <div className="w-full max-w-sm">
                        {/* Loading Animation */}
                        <div className="flex justify-center mb-8">
                          <div className="relative animate-float">
                            <div className="w-20 h-20 rounded-3xl bg-black flex items-center justify-center shadow-2xl">
                              <LoaderIcon size={28} className="text-white" />
                            </div>
                            <div className="absolute -inset-2 bg-black/10 rounded-3xl blur-2xl -z-10" />
                          </div>
                        </div>

                        {/* Progress Text */}
                        <div className="text-center mb-8">
                          <p className="text-sm font-medium">이미지를 생성하고 있습니다</p>
                          <p className="text-xs text-neutral-400 mt-1.5">첫 실행 시 모델 로딩에 시간이 걸릴 수 있습니다</p>
                          {displayedMessage && (
                            <p className="text-xs text-neutral-500 mt-3 font-mono">
                              {displayedMessage}
                              <span className="animate-pulse">|</span>
                            </p>
                          )}
                        </div>

                        {/* Tip Carousel */}
                        <div className="card-interactive p-5">
                          <div className="flex items-center gap-2 mb-3">
                            <div className="w-1.5 h-5 bg-black rounded-full" />
                            <span className="text-xs font-semibold text-neutral-500 uppercase tracking-wide">Tip</span>
                          </div>
                          <div className="min-h-[72px]">
                            <p className="text-sm font-medium mb-1.5">{TIPS[currentTip].title}</p>
                            <p className="text-xs text-neutral-500 leading-relaxed">{TIPS[currentTip].desc}</p>
                          </div>
                          {/* Dots */}
                          <div className="flex gap-2 mt-5 justify-center">
                            {TIPS.map((_, i) => (
                              <button
                                key={i}
                                onClick={() => setCurrentTip(i)}
                                className={`h-1.5 rounded-full transition-all duration-300 ${
                                  i === currentTip ? 'w-6 bg-black' : 'w-1.5 bg-neutral-200 hover:bg-neutral-300'
                                }`}
                              />
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center">
                        <div className="w-20 h-20 rounded-3xl bg-neutral-100 flex items-center justify-center mx-auto mb-5 transition-all hover:scale-105 hover:bg-neutral-200">
                          <ImageIcon size={28} className="text-neutral-300" />
                        </div>
                        <p className="text-sm text-neutral-400 font-medium">생성된 이미지가 여기에 표시됩니다</p>
                        <p className="text-xs text-neutral-300 mt-1">참조 이미지와 프롬프트를 입력하고 생성 버튼을 눌러주세요</p>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  )
}
