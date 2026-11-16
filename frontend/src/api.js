// Centralised API layer. Pages call the named helpers at the bottom and keep their own
// state, messages and rendering; nothing else in the frontend should call fetch directly.

const DEFAULT_API_BASE_URL = 'http://localhost:8000/api/v1'
const TOKEN_KEY = 'access_token'
const LOGIN_PATH = '/login'
const REGISTER_PATH = '/register'

const NETWORK_ERROR_MESSAGE = 'Unable to connect to the server.'
const GENERIC_ERROR_MESSAGE = 'Something went wrong.'
const NOT_LOGGED_IN_MESSAGE = 'Please log in to continue.'
const TOO_MANY_REQUESTS_MESSAGE = 'Too many requests. Please wait a minute and try again.'

// Configurable per environment; trailing slashes are removed so paths can start with "/".
export const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE_URL
).replace(/\/+$/, '')

// An error whose message is safe to show to the user. status is the HTTP status, or 0 when the
// server could not be reached or its response could not be read.
export class ApiError extends Error {
  constructor(message, status) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

function getToken() {
  return localStorage.getItem(TOKEN_KEY)
}

function setToken(token) {
  localStorage.setItem(TOKEN_KEY, token)
}

function clearToken() {
  localStorage.removeItem(TOKEN_KEY)
}

// Clears the (missing, stale or rejected) token and sends the user to the login page. Several
// protected requests can fail together, so this redirects at most once per page load, and never
// from the login or register page itself.
let redirecting = false

function redirectToLogin() {
  clearToken()

  const path = window.location.pathname
  if (redirecting || path === LOGIN_PATH || path === REGISTER_PATH) {
    return
  }

  redirecting = true
  window.location.assign(LOGIN_PATH)
}

// HTTP 429 always gets one friendly message: the rate limiter's body has no "detail" to show.
// Otherwise the same rules the pages used before: a string detail is shown as is, a non-empty
// array shows the first item's msg, and anything else (including an empty array or a non-JSON
// body) is generic.
function errorMessageFrom(data, status) {
  if (status === 429) {
    return TOO_MANY_REQUESTS_MESSAGE
  }

  const detail = data?.detail

  if (typeof detail === 'string' && detail.trim() !== '') {
    return detail
  }

  if (Array.isArray(detail) && typeof detail[0]?.msg === 'string' && detail[0].msg !== '') {
    return detail[0].msg
  }

  return GENERIC_ERROR_MESSAGE
}

async function request(path, { method = 'GET', body, auth = false } = {}) {
  const headers = {}

  if (auth) {
    const token = getToken()

    // Never send "Bearer null": without a token the user has to log in first.
    if (!token) {
      redirectToLogin()
      throw new ApiError(NOT_LOGGED_IN_MESSAGE, 401)
    }

    headers.Authorization = `Bearer ${token}`
  }

  let payload

  if (body instanceof FormData) {
    // Passed through untouched: the browser must set the multipart Content-Type and boundary.
    payload = body
  } else if (body !== undefined) {
    headers['Content-Type'] = 'application/json'
    payload = JSON.stringify(body)
  }

  let response
  let text

  try {
    response = await fetch(`${API_BASE_URL}${path}`, { method, headers, body: payload })
    text = await response.text()
  } catch {
    // fetch rejects only when no usable HTTP response arrived (or its body could not be read).
    throw new ApiError(NETWORK_ERROR_MESSAGE, 0)
  }

  // 204 and other empty bodies have nothing to parse; a body that is not JSON becomes null.
  let data = null

  if (text) {
    try {
      data = JSON.parse(text)
    } catch {
      data = null
    }
  }

  if (!response.ok) {
    // A rejected token on a protected request ends the session. Unauthenticated requests
    // (login, register) never get here, so a wrong password stays an ordinary error.
    if (auth && response.status === 401) {
      redirectToLogin()
    }

    throw new ApiError(errorMessageFrom(data, response.status), response.status)
  }

  // A successful response that has a body must be JSON; an empty one (204) returns null.
  if (text && data === null) {
    throw new ApiError(GENERIC_ERROR_MESSAGE, response.status)
  }

  return data
}

export async function login(email, password) {
  const data = await request('/auth/login', {
    method: 'POST',
    body: { email, password },
  })

  if (typeof data?.access_token === 'string') {
    setToken(data.access_token)
  }

  return data
}

export function register(email, password) {
  return request('/auth/register', {
    method: 'POST',
    body: { email, password },
  })
}

export function listDocuments() {
  return request('/documents', { auth: true })
}

export function getDocument(documentId) {
  return request(`/documents/${encodeURIComponent(documentId)}`, { auth: true })
}

export function createDocument(filename, description) {
  return request('/documents', {
    method: 'POST',
    body: { filename, description },
    auth: true,
  })
}

export function updateDocument(documentId, filename, description) {
  return request(`/documents/${encodeURIComponent(documentId)}`, {
    method: 'PATCH',
    body: { filename, description },
    auth: true,
  })
}

export function deleteDocument(documentId) {
  return request(`/documents/${encodeURIComponent(documentId)}`, {
    method: 'DELETE',
    auth: true,
  })
}

// Uploads one file as multipart form data. The field names (file, description) are the ones the
// backend expects. No Content-Type is set here or in request(): the browser adds the multipart
// header together with its boundary.
export function uploadDocument(file, description) {
  const form = new FormData()
  form.append('file', file)

  const trimmedDescription = description?.trim()
  if (trimmedDescription) {
    form.append('description', trimmedDescription)
  }

  return request('/documents/upload', {
    method: 'POST',
    body: form,
    auth: true,
  })
}
