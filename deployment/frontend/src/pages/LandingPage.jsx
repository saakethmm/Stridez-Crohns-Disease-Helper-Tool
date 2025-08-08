import React from 'react'
import { Link } from 'react-router-dom'

const LandingPage = () => {
  return (
    <div className="font-poppins text-gray-800 bg-gradient-to-b from-blue-100 to-white min-h-screen flex flex-col items-center">
      
      <img src="/logo.png" alt="Stridez Logo" className="h-20 mt-10 w-auto" />
      {/* App Name */}
      <h1 className="text-6xl md:text-7xl font-bold text-transparent pb-5 bg-clip-text bg-gradient-to-r from-blue-800 to-blue-400 mt-10 text-center">
        Crohnalyze
      </h1>

      {/* Welcome Message */}
      <p className="mt-6 text-xl md:text-2xl text-center font-semibold text-blue-900 px-4">
        Find peace in patterns. Predict. Track. Heal.
      </p>

      {/* Description Container */}
      <div className="mt-10 bg-white shadow-lg rounded-2xl px-6 py-8 max-w-2xl w-full mx-4">
        <h2 className="text-2xl font-bold text-blue-800 mb-4">What is Crohnalyze?</h2>
        <p className="text-gray-700 text-lg leading-relaxed">
          Crohnalyze is your personal companion for managing Crohn’s Disease. Track your symptoms, food intake, and lifestyle factors - all while leveraging AI to predict potential flare-ups. Gain clarity and control over your health journey with clean visuals, smart insights, and personalized tracking.
        </p>
      </div>

      {/* Get Started Button */}
      <Link to="/user/input">
        <button className="mt-20 bg-gradient-to-r from-blue-600 to-blue-400 text-white text-lg font-semibold px-8 py-3 rounded-full hover:from-blue-700 hover:to-blue-500 transition duration-300 animate-bounce">
            Get Started
        </button>
      </Link>

      {/* Footer */}
      <div className="mt-30 bg-blue-100 shadow-lg rounded-2xl px-2 py-2 max-w-50 w-full mx-4 text-center font-bold text-blue-900">
        <p>Made by Stridez</p>
      </div>
    </div>
  )
}

export default LandingPage