import React, { useEffect, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";

const UserOutput = () => {
  const { state } = useLocation();
  const navigate = useNavigate();
  const { id } = useParams();

  const [ingredients, setIngredients] = useState(state?.ingredients || []);
  const [advice, setAdvice] = useState(state?.advice || "");
  const [loading, setLoading] = useState(!state); // load if no state
  const [error, setError] = useState(null);

  useEffect(() => {
    // If we have state already, no need to fetch
    if (state && state.ingredients && state.advice) return;

    if (!id) {
      setError("No analysis found. Please log your food first.");
      setLoading(false);
      return;
    }

    setLoading(true);
    fetch(`http://localhost:3000/user/output/${id}`) // Your backend endpoint to fetch saved analysis by id
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch analysis data.");
        return res.json();
      })
      .then((data) => {
        if (!Array.isArray(data.ingredients) || typeof data.advice !== "string") {
          throw new Error("Invalid data received.");
        }
        setIngredients(data.ingredients);
        setAdvice(data.advice);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, [id, state]);

  if (loading) {
    return (
      <div className="text-center mt-20 text-blue-700 font-bold">
        Loading analysis...
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center mt-20">
        <p className="text-blue-800 font-bold">{error}</p>
        <button
          className="mt-4 bg-blue-700 hover:bg-blue-800 text-white px-4 py-2 rounded"
          onClick={() => navigate("/user/input")}
        >
          Back to Input
        </button>
      </div>
    );
  }

  return (
    <div className="font-poppins text-gray-800 bg-gradient-to-b from-blue-100 to-white min-h-screen p-10">
      <div className="flex justify-center">
        <img src="/logo.png" alt="Stridez Logo" className="h-20 mt-10 w-auto" />
      </div>
      <h2 className="text-xl md:text-5xl font-bold text-blue-600 text-transparent bg-clip-text bg-gradient-to-r from-blue-800 to-blue-400 mt-10 text-center pb-10">
        Your Meal Insights
      </h2>

      <div className="bg-white shadow-xl rounded-2xl p-8 max-w-3xl mx-auto mb-10">
        <h3 className="text-2xl font-bold text-blue-800 mb-4">Possible Trigger Ingredients</h3>
        <ul className="list-disc list-inside text-blue-700">
          {ingredients.map((ingredient, idx) => (
            <li key={idx}>{ingredient}</li>
          ))}
        </ul>
      </div>

      <div className="bg-white shadow-xl rounded-2xl p-8 max-w-3xl mx-auto">
        <h3 className="text-2xl font-bold text-blue-800 mb-4">Analysis and Advice</h3>
        <p className="text-blue-700 leading-relaxed">{advice}</p>
      </div>

      <div className="mt-10 text-center">
        <button
          className="bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-lg font-bold"
          onClick={() => navigate("/user/input")}
        >
          Log Another Entry
        </button>
      </div>

      <div className="mb-10 mt-10 bg-blue-100 shadow-lg rounded-2xl px-4 py-2 max-w-50 mx-auto text-center font-bold text-blue-900">
        <p>Made by Stridez</p>
      </div>
    </div>
  );
};

export default UserOutput;
