import { useEffect, useRef, useState } from 'react'
import {
  createDocument,
  deleteDocument,
  getDocument,
  listDocuments,
  updateDocument,
  uploadDocument,
} from './api'

// These mirror the backend (settings.allowed_extensions and max_upload_mb) so obvious mistakes are
// caught before uploading. The backend still validates everything and has the final say.
const ALLOWED_UPLOAD_EXTENSIONS = ['.pdf', '.txt', '.md', '.docx']
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024

function validateUploadFile(file) {
  if (!file) {
    return 'Please choose a file.'
  }

  // Like the backend, a name with no dot after its first character has no extension.
  const dotIndex = file.name.lastIndexOf('.')
  const extension = dotIndex > 0 ? file.name.slice(dotIndex).toLowerCase() : ''

  if (!ALLOWED_UPLOAD_EXTENSIONS.includes(extension)) {
    return 'Unsupported file type. Allowed: PDF, TXT, MD, DOCX.'
  }

  if (file.size > MAX_UPLOAD_BYTES) {
    return 'File is too large (maximum 10 MB).'
  }

  if (file.size === 0) {
    return 'The selected file is empty. Please choose a file with content.'
  }

  return null
}

// Extraction statuses the backend can store (documents.processing_status). A missing (null) status
// belongs to a metadata-only document that is never processed, so it shows nothing and never polls.
const STATUS_LABELS = {
  pending: 'Pending',
  processing: 'Processing',
  completed: 'Completed',
  failed: 'Failed',
}
const ACTIVE_STATUSES = ['pending', 'processing']

const POLL_INTERVAL_MS = 3000
const POLL_MAX_DURATION_MS = 5 * 60 * 1000
const POLL_MAX_CONSECUTIVE_FAILURES = 3

const POLL_CAP_NOTE = 'Status updates paused after five minutes. Refresh the page to check again.'
const POLL_FAILURE_NOTE = 'Some document statuses could not be updated. Refresh the page to check again.'

function isActiveStatus(status) {
  return ACTIVE_STATUSES.includes(status)
}

// Shows the extraction status of one document and, for a failed one, why it failed.
function ProcessingStatus({ status, error }) {
  if (status === null || status === undefined) {
    return null
  }

  // Object.hasOwn keeps inherited names such as "constructor" from being read as a status.
  const label = Object.hasOwn(STATUS_LABELS, status) ? STATUS_LABELS[status] : 'Unknown'
  const showError = status === 'failed' && typeof error === 'string' && error.trim() !== ''

  return (
    <>
      <p>Status: {label}</p>
      {showError && <p>Processing failed: {error}</p>}
    </>
  )
}

function Documents() {
  const [documents, setDocuments] = useState([])

  const [loading, setLoading] = useState(true)
  const [creating, setCreating] = useState(false)
  const [updating, setUpdating] = useState(false)
  const [deleting, setDeleting] = useState(false)

  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const [filename, setFilename] = useState('')
  const [description, setDescription] = useState('')

  const [editingDocument, setEditingDocument] = useState(null)
  const [editFilename, setEditFilename] = useState('')
  const [editDescription, setEditDescription] = useState('')

  const [uploading, setUploading] = useState(false)
  const [file, setFile] = useState(null)
  const [uploadDescription, setUploadDescription] = useState('')
  const [uploadError, setUploadError] = useState('')
  const [fileInputKey, setFileInputKey] = useState(0)

  // Ids polling has given up on (three failures, a 404, or the time cap). They stay as they are
  // until the page is refreshed, so the polling effect does not start over for them.
  const [stoppedPollingIds, setStoppedPollingIds] = useState([])
  const [statusNote, setStatusNote] = useState('')

  // Not state: nothing renders from these. pollStartRef is when the current polling run began (it
  // survives the effect restarting, so a status change does not restart the five minutes),
  // failureCountsRef maps document id to consecutive failures, inFlightRef holds ids with a
  // request still running so the same document is never requested twice at once.
  const pollStartRef = useRef(null)
  const failureCountsRef = useRef(new Map())
  const inFlightRef = useRef(new Set())

  // Documents that still need polling: extraction not finished and not given up on. The key is a
  // string so the effect below restarts only when this set changes, not on every render.
  const activePollingKey = documents
    .filter(
      (document) =>
        isActiveStatus(document.processing_status) &&
        !stoppedPollingIds.includes(document.id)
    )
    .map((document) => document.id)
    .join(',')

  useEffect(() => {
        const fetchDocuments = async () => {
            setLoading(true)

            try {
            const data = await listDocuments()

            setDocuments(data)
            setError('')
            } catch (err) {
            setError(err.message)
            } finally {
            setLoading(false)
            }
        }

        fetchDocuments()
    }, [])

  useEffect(() => {
    if (activePollingKey === '') {
      // Nothing to poll: the next document that becomes active starts a fresh five minutes.
      pollStartRef.current = null
      return undefined
    }

    if (pollStartRef.current === null) {
      pollStartRef.current = Date.now()
    }

    const ids = activePollingKey.split(',').map(Number)
    let cancelled = false
    let timerId = null

    const stopPolling = (idsToStop) => {
      setStoppedPollingIds((previousIds) => [...new Set([...previousIds, ...idsToStop])])
    }

    const poll = async () => {
      if (Date.now() - pollStartRef.current >= POLL_MAX_DURATION_MS) {
        stopPolling(ids)
        setStatusNote(POLL_CAP_NOTE)
        return
      }

      const stoppedNow = new Set()

      // One document at a time, so at most one polling request is ever in flight per tick.
      for (const id of ids) {
        if (cancelled) {
          return
        }

        if (inFlightRef.current.has(id)) {
          continue
        }

        inFlightRef.current.add(id)

        try {
          const data = await getDocument(id)

          if (cancelled) {
            return
          }

          failureCountsRef.current.delete(id)

          // Only the status fields are copied into the existing row. Replacing the whole row
          // could overwrite an edit saved while this request was running.
          setDocuments((previousDocuments) =>
            previousDocuments.map((document) =>
              document.id === id
                ? {
                    ...document,
                    processing_status: data.processing_status,
                    processing_error: data.processing_error,
                  }
                : document
            )
          )
        } catch (err) {
          if (cancelled) {
            return
          }

          if (err.status === 401) {
            // api.js has already cleared the token and is redirecting to the login page.
            return
          }

          if (err.status === 404) {
            // The document is gone on the server; keep it displayed but stop asking about it.
            failureCountsRef.current.delete(id)
            stoppedNow.add(id)
          } else {
            const failures = (failureCountsRef.current.get(id) ?? 0) + 1
            failureCountsRef.current.set(id, failures)

            if (failures >= POLL_MAX_CONSECUTIVE_FAILURES) {
              failureCountsRef.current.delete(id)
              stoppedNow.add(id)
              setStatusNote(POLL_FAILURE_NOTE)
            }
          }
        } finally {
          inFlightRef.current.delete(id)
        }
      }

      if (stoppedNow.size > 0) {
        stopPolling([...stoppedNow])
      }

      // The next tick is scheduled only after this one has finished, and only if some
      // document is still being polled (stopping others restarts this effect anyway).
      if (!cancelled && stoppedNow.size < ids.length) {
        timerId = setTimeout(poll, POLL_INTERVAL_MS)
      }
    }

    timerId = setTimeout(poll, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      clearTimeout(timerId)
    }
  }, [activePollingKey])

  const handleCreateDocument = async (event) => {
  event.preventDefault()

  setCreating(true)

  try {
    const data = await createDocument(filename, description)

    setDocuments((previousDocuments) => [
      ...previousDocuments,
      data,
    ])

    setFilename('')
    setDescription('')
    setSuccess('Document created successfully!')
    setError('')
  } catch (err) {
    setError(err.message)
  } finally {
    setCreating(false)
  }
   }

  const handleUploadDocument = async (event) => {
    event.preventDefault()

    setSuccess('')
    setUploadError('')

    const validationError = validateUploadFile(file)

    if (validationError) {
      setUploadError(validationError)
      return
    }

    setUploading(true)

    try {
      const data = await uploadDocument(file, uploadDescription)

      setDocuments((previousDocuments) => [
        ...previousDocuments,
        data,
      ])

      setFile(null)
      setUploadDescription('')
      setFileInputKey((previousKey) => previousKey + 1)
      setSuccess('Document uploaded successfully!')
    } catch (err) {
      setUploadError(err.message)
    } finally {
      setUploading(false)
    }
  }

  const handleEdit = (document) => {
    setEditingDocument(document)
    setEditFilename(document.filename)
    setEditDescription(document.description || '')
   }

  const handleUpdateDocument = async (event) => {
        event.preventDefault()

        setUpdating(true)
        try {
            const data = await updateDocument(
                editingDocument.id,
                editFilename,
                editDescription
            )

            setDocuments((previousDocuments) =>
                previousDocuments.map((document) =>
                document.id === data.id ? data : document
                )
            )

            setEditingDocument(null)
            setSuccess('Document updated successfully!')
            setError('')
        } catch (err) {
            setError(err.message)
        } finally {
            setUpdating(false)
        }
  }

  const handleDeleteDocument = async (documentId) => {
  const confirmed = window.confirm(
    'Are you sure you want to delete this document?'
  )

  if (!confirmed) {
    return
  }

  setDeleting(true)

  try {
    await deleteDocument(documentId)

    setDocuments((previousDocuments) =>
      previousDocuments.filter(
        (document) => document.id !== documentId
      )
    )

    setError('')
  } catch (err) {
    setError(err.message)
  } finally {
    setDeleting(false)
  }
}

  return (
  <div>
    <h1>My Documents</h1>

    {statusNote && <p>{statusNote}</p>}

    <h2>Create Document</h2>

    <form onSubmit={handleCreateDocument}>
        <div>
            <label htmlFor="filename">Filename</label>

            <input
            id="filename"
            type="text"
            value={filename}
            onChange={(event) => setFilename(event.target.value)}
            />
        </div>

        <div>
            <label htmlFor="description">Description</label>

            <input
            id="description"
            type="text"
            value={description}
            onChange={(event) => setDescription(event.target.value)}
            />
        </div>

        <button type="submit" disabled={creating}>
                {creating ? 'Creating...' : 'Create Document'}
        </button>
    </form>

    <h2>Upload Document</h2>

    <form onSubmit={handleUploadDocument}>
        <div>
            <label htmlFor="upload-file">File</label>

            <input
            key={fileInputKey}
            id="upload-file"
            type="file"
            accept=".pdf,.txt,.md,.docx"
            disabled={uploading}
            onChange={(event) => {
                setFile(event.target.files[0] ?? null)
                setUploadError('')
            }}
            />
        </div>

        <div>
            <label htmlFor="upload-description">Description</label>

            <input
            id="upload-description"
            type="text"
            maxLength={1000}
            value={uploadDescription}
            disabled={uploading}
            onChange={(event) => setUploadDescription(event.target.value)}
            />
        </div>

        <button type="submit" disabled={uploading}>
                {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
    </form>

    {uploadError && <p>{uploadError}</p>}

    {loading && <p>Loading documents...</p>}
    {success && <p>{success}</p>}
    {error && <p>{error}</p>}

    {!loading && !error && (
      <div>
        {documents.map((document) => (
          <div key={document.id}>

            <h2>{document.filename}</h2>
            <p>{document.description}</p>

            <ProcessingStatus
              status={document.processing_status}
              error={document.processing_error}
            />

            <button onClick={() => handleEdit(document)}>
                Edit
            </button>

            <button
                onClick={() => handleDeleteDocument(document.id)}
                disabled={deleting}
                >
                {deleting ? 'Deleting...' : 'Delete'}
            </button>

            {editingDocument?.id === document.id && (
                <div>
                    <h3>Edit Document</h3>

                    <form onSubmit={handleUpdateDocument}>
                    <div>
                        <label htmlFor="edit-filename">Filename</label>

                        <input
                        id="edit-filename"
                        type="text"
                        value={editFilename}
                        onChange={(event) => setEditFilename(event.target.value)}
                        />
                    </div>

                    <div>
                        <label htmlFor="edit-description">Description</label>

                        <input
                        id="edit-description"
                        type="text"
                        value={editDescription}
                        onChange={(event) => setEditDescription(event.target.value)}
                        />
                    </div>

                    <button type="submit" disabled={updating}>
                        {updating ? 'Saving...' : 'Save'}
                    </button>

                    <button
                        type="button"
                        onClick={() => setEditingDocument(null)}
                    >
                        Cancel
                    </button>
                    </form>
                </div>
                )}

          </div>
        ))}
      </div>
    )}
  </div>
)
}

export default Documents