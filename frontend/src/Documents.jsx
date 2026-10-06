import { useEffect, useState } from 'react'

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

  useEffect(() => {
        const fetchDocuments = async () => {
            setLoading(true)

            try {
            const token = localStorage.getItem('access_token')

            const response = await fetch(
                'http://localhost:8000/api/v1/documents',
                {
                method: 'GET',
                headers: {
                    Authorization: `Bearer ${token}`,
                },
                }
            )

            const data = await response.json()

            if (response.ok) {
                setDocuments(data)
                setError('')
            } else {
                if (typeof data.detail === 'string') {
                setError(data.detail)
                } else if (Array.isArray(data.detail)) {
                setError(data.detail[0].msg)
                } else {
                setError('Something went wrong.')
                }
            }
            } catch (error) {
            setError('Unable to connect to the server.')
            } finally {
            setLoading(false)
            }
        }

        fetchDocuments()
    }, [])

  const handleCreateDocument = async (event) => {
  event.preventDefault()

  setCreating(true)

  try {
    const token = localStorage.getItem('access_token')

    const response = await fetch(
      'http://localhost:8000/api/v1/documents',
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          filename: filename,
          description: description,
        }),
      }
    )

    const data = await response.json()

    if (response.ok) {
      setDocuments((previousDocuments) => [
        ...previousDocuments,
        data,
      ])

      setFilename('')
      setDescription('')
      setSuccess('Document created successfully!')
      setError('')
    } else {
      if (typeof data.detail === 'string') {
        setError(data.detail)
      } else if (Array.isArray(data.detail)) {
        setError(data.detail[0].msg)
      } else {
        setError('Something went wrong.')
      }
    }
  } catch (error) {
    setError('Unable to connect to the server.')
  } finally {
    setCreating(false)
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
            const token = localStorage.getItem('access_token')

            const response = await fetch(
            `http://localhost:8000/api/v1/documents/${editingDocument.id}`,
            {
                method: 'PATCH',
                headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${token}`,
                },
                body: JSON.stringify({
                filename: editFilename,
                description: editDescription,
                }),
            }
            )

            const data = await response.json()

            if (response.ok) {
            setDocuments((previousDocuments) =>
                previousDocuments.map((document) =>
                document.id === data.id ? data : document
                )
            )

            setEditingDocument(null)
            setSuccess('Document updated successfully!')
            setError('')
            } else {
            if (typeof data.detail === 'string') {
                setError(data.detail)
            } else if (Array.isArray(data.detail)) {
                setError(data.detail[0].msg)
            } else {
                setError('Something went wrong.')
            }
            }
        } catch (error) {
            setError('Unable to connect to the server.')
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
    const token = localStorage.getItem('access_token')

    const response = await fetch(
      `http://localhost:8000/api/v1/documents/${documentId}`,
      {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      }
    )

    if (response.ok) {
      setDocuments((previousDocuments) =>
        previousDocuments.filter(
          (document) => document.id !== documentId
        )
      )

      setError('')
    } else {
      const data = await response.json()

      if (typeof data.detail === 'string') {
        setError(data.detail)
      } else if (Array.isArray(data.detail)) {
        setError(data.detail[0].msg)
      } else {
        setError('Something went wrong.')
      }
    }
  } catch (error) {
    setError('Unable to connect to the server.')
  } finally {
    setDeleting(false)
  }
}

  return (
  <div>
    <h1>My Documents</h1>

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

    {loading && <p>Loading documents...</p>}
    {success && <p>{success}</p>}
    {error && <p>{error}</p>}

    {!loading && !error && (
      <div>
        {documents.map((document) => (
          <div key={document.id}>

            <h2>{document.filename}</h2>
            <p>{document.description}</p>

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