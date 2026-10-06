import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useNavigate } from 'react-router-dom'

function Login() {
  const navigate = useNavigate()

  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  const handleSubmit = async (event) => {
        event.preventDefault()

        setLoading(true)

        try {
            const response = await fetch(
            'http://localhost:8000/api/v1/auth/login',
            {
                method: 'POST',
                headers: {
                'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                email: email,
                password: password,
                }),
            }
            )

            const data = await response.json()

            if (response.ok) {
            localStorage.setItem('access_token', data.access_token)

            setSuccess('Login successful!')
            setError('')

            navigate('/documents')
            } 
            else {
            if (typeof data.detail === 'string') {
                setError(data.detail)
            } else if (Array.isArray(data.detail)) {
                setError(data.detail[0].msg)
            } else {
                setError('Something went wrong.')
            }
            setSuccess('')
            }
        } 
        catch (error) {
            setError('Unable to connect to the server.')
            setSuccess('')
        } 
        finally {
            setLoading(false)
        }
        console.log('Status:', response.status)
        console.log('Response:', data)
  }
       
  return (
    <div>
      <h1>Login to DocuMind</h1>

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
            {loading ? 'Logging in...' : 'Login'}
        </button>
      </form>
      <p>
        Don't have an account? <Link to="/register">Register</Link>
      </p>
    </div>
  )
}

export default Login