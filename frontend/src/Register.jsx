import { useState } from 'react'
import { register } from './api'
import { Link } from 'react-router-dom'

function Register() {

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')
  const [loading, setLoading] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()

    setLoading(true)

    try {
      await register(email, password)

      setSuccess('Account created successfully!')
      setError('')
      setEmail('')
      setPassword('')
    }
    catch (err) {
      setError(err.message)
      setSuccess('')
    }
    finally {
    setLoading(false)
    }
}

  return (
    <div>
      <h1>Create your DocuMind account</h1>

      {error && <p>{error}</p>}

      {success && <p>{success}</p>}

      <form onSubmit={handleSubmit}>
        <div>
          <label htmlFor="email">Email</label>
          <input
            id="email"
            type="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>

        <div>
          <label htmlFor="password">Password</label>
          <input
            id="password"
            type="password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        <button type="submit" disabled={loading}>
            {loading ? 'Creating account...' : 'Register'}
        </button>
      </form>
      <p>
        Already have an account? <Link to="/login">Login</Link>
      </p>
    </div>
  )
}

export default Register