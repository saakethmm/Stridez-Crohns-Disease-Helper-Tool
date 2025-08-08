import React from 'react'
import {Routes, Route, Router} from 'react-router-dom'
import LandingPage from './pages/LandingPage'
import UserOutput from './pages/UserOutput'
import UserInput from './pages/UserInput'

const App = () => {
  return (
    <Routes>
      <Route path='/' element={<LandingPage />} />
      <Route path='/user/input' element={<UserInput />} />
      <Route path='/user/output/:id' element={<UserOutput />} />
    </Routes>
  )
}

export default App