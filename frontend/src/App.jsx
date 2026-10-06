import { BrowserRouter, Routes, Route } from 'react-router-dom'

import Register from './Register'
import Login from './Login'
import Documents from './Documents'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/register" element={<Register />} />
        <Route path="/login" element={<Login />} />
        <Route path="/documents" element={<Documents />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App