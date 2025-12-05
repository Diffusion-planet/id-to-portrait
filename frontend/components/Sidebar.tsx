'use client'

import { useState, useRef, useCallback } from 'react'
import {
  MenuIcon,
  FolderIcon,
  FolderOpenIcon,
  PlusIcon,
  TrashIcon,
  EditIcon,
  ClockIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MoreIcon,
  XIcon,
  CheckIcon,
} from './Icons'
import { HistoryItem, Folder, getImageUrl } from '@/lib/api'

interface SidebarProps {
  isOpen: boolean
  onToggle: () => void
  history: HistoryItem[]
  folders: Folder[]
  selectedHistoryId: string | null
  onSelectHistory: (item: HistoryItem) => void
  onDeleteHistory: (id: string) => void
  onUpdateHistoryTitle: (id: string, title: string) => void
  onMoveToFolder: (historyId: string, folderId: string | null) => void
  onCreateFolder: (name: string) => void
  onDeleteFolder: (id: string) => void
  onRenameFolder: (id: string, name: string) => void
}

export default function Sidebar({
  isOpen,
  onToggle,
  history,
  folders,
  selectedHistoryId,
  onSelectHistory,
  onDeleteHistory,
  onUpdateHistoryTitle,
  onMoveToFolder,
  onCreateFolder,
  onDeleteFolder,
  onRenameFolder,
}: SidebarProps) {
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(new Set())
  const [editingHistoryId, setEditingHistoryId] = useState<string | null>(null)
  const [editingFolderId, setEditingFolderId] = useState<string | null>(null)
  const [editValue, setEditValue] = useState('')
  const [showNewFolder, setShowNewFolder] = useState(false)
  const [newFolderName, setNewFolderName] = useState('')
  const [draggedItem, setDraggedItem] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<string | null>(null)
  const [contextMenu, setContextMenu] = useState<{
    type: 'history' | 'folder'
    id: string
    x: number
    y: number
  } | null>(null)

  const inputRef = useRef<HTMLInputElement>(null)

  const toggleFolder = (folderId: string) => {
    setExpandedFolders((prev) => {
      const next = new Set(prev)
      if (next.has(folderId)) {
        next.delete(folderId)
      } else {
        next.add(folderId)
      }
      return next
    })
  }

  const startEditHistory = (id: string, currentTitle: string) => {
    setEditingHistoryId(id)
    setEditValue(currentTitle)
    setContextMenu(null)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  const startEditFolder = (id: string, currentName: string) => {
    setEditingFolderId(id)
    setEditValue(currentName)
    setContextMenu(null)
    setTimeout(() => inputRef.current?.focus(), 0)
  }

  const saveEdit = () => {
    if (editingHistoryId && editValue.trim()) {
      onUpdateHistoryTitle(editingHistoryId, editValue.trim())
    }
    if (editingFolderId && editValue.trim()) {
      onRenameFolder(editingFolderId, editValue.trim())
    }
    setEditingHistoryId(null)
    setEditingFolderId(null)
    setEditValue('')
  }

  const cancelEdit = () => {
    setEditingHistoryId(null)
    setEditingFolderId(null)
    setEditValue('')
  }

  const handleCreateFolder = () => {
    if (newFolderName.trim()) {
      onCreateFolder(newFolderName.trim())
      setNewFolderName('')
      setShowNewFolder(false)
    }
  }

  // Drag and drop handlers
  const handleDragStart = (e: React.DragEvent, historyId: string) => {
    setDraggedItem(historyId)
    e.dataTransfer.effectAllowed = 'move'
  }

  const handleDragOver = (e: React.DragEvent, targetId: string | null) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setDropTarget(targetId)
  }

  const handleDragLeave = () => {
    setDropTarget(null)
  }

  const handleDrop = (e: React.DragEvent, folderId: string | null) => {
    e.preventDefault()
    if (draggedItem) {
      onMoveToFolder(draggedItem, folderId)
    }
    setDraggedItem(null)
    setDropTarget(null)
  }

  const handleContextMenu = (
    e: React.MouseEvent,
    type: 'history' | 'folder',
    id: string
  ) => {
    e.preventDefault()
    setContextMenu({ type, id, x: e.clientX, y: e.clientY })
  }

  const closeContextMenu = () => setContextMenu(null)

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / 60000)
    const diffHours = Math.floor(diffMs / 3600000)
    const diffDays = Math.floor(diffMs / 86400000)

    if (diffMins < 1) return '방금 전'
    if (diffMins < 60) return `${diffMins}분 전`
    if (diffHours < 24) return `${diffHours}시간 전`
    if (diffDays < 7) return `${diffDays}일 전`
    return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' })
  }

  // Group history by folder
  const unfiledHistory = history.filter((h) => !h.folder_id)
  const historyByFolder = folders.reduce((acc, folder) => {
    acc[folder.id] = history.filter((h) => h.folder_id === folder.id)
    return acc
  }, {} as Record<string, HistoryItem[]>)

  const renderHistoryItem = (item: HistoryItem) => {
    const isEditing = editingHistoryId === item.id
    const isSelected = selectedHistoryId === item.id
    const isDragging = draggedItem === item.id

    return (
      <div
        key={item.id}
        draggable={!isEditing}
        onDragStart={(e) => handleDragStart(e, item.id)}
        onContextMenu={(e) => handleContextMenu(e, 'history', item.id)}
        onClick={() => !isEditing && onSelectHistory(item)}
        className={`
          group relative px-3 py-2.5 rounded-xl cursor-pointer
          transition-all duration-200 ease-out
          ${isSelected ? 'bg-black text-white' : 'hover:bg-neutral-100'}
          ${isDragging ? 'opacity-50' : ''}
        `}
      >
        <div className="flex items-start gap-3">
          <div className="w-10 h-10 rounded-lg overflow-hidden flex-shrink-0 bg-neutral-100">
            <img
              src={getImageUrl(item.output_image_url)}
              alt=""
              className="w-full h-full object-cover"
            />
          </div>
          <div className="flex-1 min-w-0">
            {isEditing ? (
              <div className="flex items-center gap-1">
                <input
                  ref={inputRef}
                  type="text"
                  value={editValue}
                  onChange={(e) => setEditValue(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') saveEdit()
                    if (e.key === 'Escape') cancelEdit()
                  }}
                  onBlur={saveEdit}
                  className="w-full px-2 py-0.5 text-sm bg-white text-black rounded border border-neutral-300 focus:outline-none focus:border-black"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
            ) : (
              <>
                <p className="text-sm font-medium truncate">{item.title}</p>
                <p
                  className={`text-xs mt-0.5 ${
                    isSelected ? 'text-neutral-300' : 'text-neutral-500'
                  }`}
                >
                  {formatDate(item.created_at)}
                </p>
              </>
            )}
          </div>
          {!isEditing && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleContextMenu(e, 'history', item.id)
              }}
              className={`
                opacity-0 group-hover:opacity-100 p-1 rounded-md
                transition-opacity duration-150
                ${isSelected ? 'hover:bg-white/20' : 'hover:bg-neutral-200'}
              `}
            >
              <MoreIcon size={16} />
            </button>
          )}
        </div>
      </div>
    )
  }

  const renderFolder = (folder: Folder) => {
    const isExpanded = expandedFolders.has(folder.id)
    const isEditing = editingFolderId === folder.id
    const folderItems = historyByFolder[folder.id] || []
    const isDropTarget = dropTarget === folder.id

    return (
      <div key={folder.id} className="mb-1">
        <div
          onDragOver={(e) => handleDragOver(e, folder.id)}
          onDragLeave={handleDragLeave}
          onDrop={(e) => handleDrop(e, folder.id)}
          onContextMenu={(e) => handleContextMenu(e, 'folder', folder.id)}
          className={`
            group flex items-center gap-2 px-3 py-2 rounded-xl cursor-pointer
            transition-all duration-200 ease-out
            ${isDropTarget ? 'bg-neutral-200 ring-2 ring-black ring-inset' : 'hover:bg-neutral-100'}
          `}
          onClick={() => toggleFolder(folder.id)}
        >
          {isExpanded ? (
            <FolderOpenIcon size={18} className="text-neutral-600" />
          ) : (
            <FolderIcon size={18} className="text-neutral-600" />
          )}
          {isEditing ? (
            <input
              ref={inputRef}
              type="text"
              value={editValue}
              onChange={(e) => setEditValue(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') saveEdit()
                if (e.key === 'Escape') cancelEdit()
              }}
              onBlur={saveEdit}
              className="flex-1 px-2 py-0.5 text-sm bg-white rounded border border-neutral-300 focus:outline-none focus:border-black"
              onClick={(e) => e.stopPropagation()}
            />
          ) : (
            <span className="flex-1 text-sm font-medium truncate">
              {folder.name}
            </span>
          )}
          <span className="text-xs text-neutral-400">{folderItems.length}</span>
          {!isEditing && (
            <button
              onClick={(e) => {
                e.stopPropagation()
                handleContextMenu(e, 'folder', folder.id)
              }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded-md hover:bg-neutral-200 transition-opacity duration-150"
            >
              <MoreIcon size={16} />
            </button>
          )}
        </div>
        {isExpanded && folderItems.length > 0 && (
          <div className="ml-4 mt-1 space-y-1 border-l-2 border-neutral-100 pl-2">
            {folderItems.map(renderHistoryItem)}
          </div>
        )}
      </div>
    )
  }

  return (
    <>
      {/* Toggle button when closed - positioned below header */}
      {!isOpen && (
        <button
          onClick={onToggle}
          className="fixed left-4 top-20 z-40 p-2.5 bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 ease-out hover:scale-105 active:scale-95"
        >
          <MenuIcon size={20} />
        </button>
      )}

      {/* Sidebar panel */}
      <div
        className={`
          fixed left-0 top-0 h-full bg-white z-50
          transition-all duration-300 ease-out
          shadow-2xl
          ${isOpen ? 'w-80 translate-x-0' : 'w-80 -translate-x-full'}
        `}
        onClick={closeContextMenu}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b border-neutral-100">
          <div className="flex items-center gap-2">
            <ClockIcon size={20} className="text-neutral-600" />
            <h2 className="font-semibold">히스토리</h2>
          </div>
          <button
            onClick={onToggle}
            className="p-2 rounded-xl hover:bg-neutral-100 transition-colors duration-150"
          >
            <ChevronLeftIcon size={20} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto h-[calc(100%-130px)] px-2 py-3">
          {/* Folders */}
          {folders.length > 0 && (
            <div className="mb-4">
              <div className="flex items-center justify-between px-3 py-2">
                <span className="text-xs font-medium text-neutral-400 uppercase tracking-wider">
                  폴더
                </span>
              </div>
              {folders
                .sort((a, b) => a.order - b.order)
                .map(renderFolder)}
            </div>
          )}

          {/* Unfiled history */}
          <div>
            <div
              className={`
                flex items-center justify-between px-3 py-2 rounded-xl
                transition-all duration-200
                ${dropTarget === 'unfiled' ? 'bg-neutral-200 ring-2 ring-black ring-inset' : ''}
              `}
              onDragOver={(e) => handleDragOver(e, 'unfiled')}
              onDragLeave={handleDragLeave}
              onDrop={(e) => handleDrop(e, null)}
            >
              <span className="text-xs font-medium text-neutral-400 uppercase tracking-wider">
                최근 기록
              </span>
              <span className="text-xs text-neutral-400">
                {unfiledHistory.length}
              </span>
            </div>
            <div className="space-y-1 mt-1">
              {unfiledHistory.map(renderHistoryItem)}
            </div>
          </div>

          {history.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-neutral-400">
              <ClockIcon size={32} className="mb-2 opacity-50" />
              <p className="text-sm">아직 기록이 없습니다</p>
            </div>
          )}
        </div>

        {/* Footer - New Folder */}
        <div className="absolute bottom-0 left-0 right-0 px-4 py-3 border-t border-neutral-100 bg-white">
          {showNewFolder ? (
            <div className="flex items-center gap-2">
              <input
                type="text"
                value={newFolderName}
                onChange={(e) => setNewFolderName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleCreateFolder()
                  if (e.key === 'Escape') {
                    setShowNewFolder(false)
                    setNewFolderName('')
                  }
                }}
                placeholder="폴더 이름"
                className="flex-1 px-3 py-2 text-sm bg-neutral-100 rounded-lg focus:outline-none focus:ring-2 focus:ring-black"
                autoFocus
              />
              <button
                onClick={handleCreateFolder}
                className="p-2 bg-black text-white rounded-lg hover:bg-neutral-800 transition-colors"
              >
                <CheckIcon size={16} />
              </button>
              <button
                onClick={() => {
                  setShowNewFolder(false)
                  setNewFolderName('')
                }}
                className="p-2 hover:bg-neutral-100 rounded-lg transition-colors"
              >
                <XIcon size={16} />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowNewFolder(true)}
              className="flex items-center gap-2 w-full px-3 py-2 text-sm text-neutral-600 hover:bg-neutral-100 rounded-lg transition-colors"
            >
              <PlusIcon size={16} />
              <span>새 폴더</span>
            </button>
          )}
        </div>
      </div>

      {/* Context Menu */}
      {contextMenu && (
        <>
          <div
            className="fixed inset-0 z-[60]"
            onClick={closeContextMenu}
          />
          <div
            className="fixed z-[70] bg-white rounded-xl shadow-xl border border-neutral-100 py-1 min-w-[140px]"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            {contextMenu.type === 'history' ? (
              <>
                <button
                  onClick={() => {
                    const item = history.find((h) => h.id === contextMenu.id)
                    if (item) startEditHistory(item.id, item.title)
                  }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm hover:bg-neutral-100 transition-colors"
                >
                  <EditIcon size={14} />
                  <span>이름 변경</span>
                </button>
                <button
                  onClick={() => {
                    onDeleteHistory(contextMenu.id)
                    closeContextMenu()
                  }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                >
                  <TrashIcon size={14} />
                  <span>삭제</span>
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={() => {
                    const folder = folders.find((f) => f.id === contextMenu.id)
                    if (folder) startEditFolder(folder.id, folder.name)
                  }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm hover:bg-neutral-100 transition-colors"
                >
                  <EditIcon size={14} />
                  <span>이름 변경</span>
                </button>
                <button
                  onClick={() => {
                    onDeleteFolder(contextMenu.id)
                    closeContextMenu()
                  }}
                  className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 hover:bg-red-50 transition-colors"
                >
                  <TrashIcon size={14} />
                  <span>삭제</span>
                </button>
              </>
            )}
          </div>
        </>
      )}

      {/* Backdrop when sidebar is open */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/20 z-40 transition-opacity duration-300"
          onClick={onToggle}
        />
      )}
    </>
  )
}
