import React, { useState, useEffect } from "react";
import Papa from "papaparse";

// List of symptoms available for selection
const symptomsList = [
  "Abdominal pain",
  "Diarrhea",
  "Fatigue",
  "Bloating",
  "Joint pain",
  "Mouth sores",
  "Fever",
  "Constipation",
  "Nausea",
];

const UserInput = () => {
  // --- State hooks ---
  const [options, setOptions] = useState([]); // Food options from CSV
  const [selectedFood, setSelectedFood] = useState(""); // Selected food name
  const [foodDate, setFoodDate] = useState(""); // Food consumed datetime
  const [selectedSymptoms, setSelectedSymptoms] = useState({}); // Symptoms + severity
  const [symptomDate, setSymptomDate] = useState(""); // Symptom occurred datetime
  const [savedLogs, setSavedLogs] = useState([]); // Saved logs to display

  // --- Load food options from CSV on mount ---
  useEffect(() => {
    fetch("/ingredients.csv")
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error: ${res.status}`);
        return res.text();
      })
      .then((text) => {
        const parsed = Papa.parse(text, { header: true });
        const cleanedData = parsed.data
          .map((row) => row.Description)
          .filter(Boolean);
        setOptions(cleanedData);
      })
      .catch((err) => {
        console.error("Error loading CSV:", err);
      });
  }, []);

  // --- Handlers for symptom selection and severity changes ---
  const handleSymptomToggle = (symptom) => {
    setSelectedSymptoms((prev) => ({
      ...prev,
      [symptom]: prev[symptom] ? undefined : 1, // toggle, default severity 1
    }));
  };

  const handleSeverityChange = (symptom, value) => {
    setSelectedSymptoms((prev) => ({
      ...prev,
      [symptom]: value,
    }));
  };

  // --- Form submission handler ---
  const handleSubmit = async () => {
    if (!selectedFood) {
      alert("Please select a food item.");
      return;
    }

    const selectedSymptomsArray = Object.entries(selectedSymptoms).filter(
      ([_, severity]) => severity !== undefined
    );

    if (selectedSymptomsArray.length === 0) {
      alert("Please select at least one symptom.");
      return;
    }

    const dishes = [
      {
        name: selectedFood,
        consumedAt: foodDate || new Date().toISOString(),
      },
    ];

    const symptoms = selectedSymptomsArray.map(([name, severity]) => ({
      symptom: name,
      severity,
      occurredAt: symptomDate || new Date().toISOString(),
    }));

    const payload = {
      userId: null, // adjust if track users
      dishes,
      symptoms,
      submittedAt: new Date().toISOString(),
    };

    try {
      const res = await fetch("http://localhost:3000/user/input", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errorData = await res.json();
        alert("Error: " + errorData.error);
        return;
      }

      const data = await res.json();
      alert("Log saved successfully!");
      setSavedLogs((prev) => [data, ...prev]);

      // Reset form fields
      setSelectedFood("");
      setFoodDate("");
      setSelectedSymptoms({});
      setSymptomDate("");
    } catch (err) {
      console.error("Submission error:", err);
      alert("Failed to submit data.");
    }
  };

  // --- JSX return ---
  return (
    <div className="font-poppins text-gray-800 bg-gradient-to-b from-blue-100 to-white min-h-screen flex flex-col items-center">
      {/* Header */}
      <h2 className="text-xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-800 to-blue-400 mt-20 text-center pb-20 pt-10">
        Remissi, Your Personal Companion <br /> Log Your Food and Symptoms Here!
      </h2>

      {/* Food Log Section */}
      <div className="bg-white shadow-lg rounded-2xl px-6 py-6 w-full max-w-2xl mb-8">
        <h2 className="text-2xl font-bold text-blue-800 mb-4">Your Food Log</h2>
        <div className="flex flex-col md:flex-row items-center gap-4">
          <select
            className="border border-blue-700 rounded px-3 py-2 text-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 w-full md:w-1/2"
            value={selectedFood}
            onChange={(e) => setSelectedFood(e.target.value)}
          >
            <option value="">Select Food</option>
            {options.map((food, idx) => (
              <option key={idx} value={food}>
                {food}
              </option>
            ))}
          </select>

          <input
            type="datetime-local"
            className="border border-blue-700 rounded px-3 py-2 text-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-400 w-full md:w-1/2"
            value={foodDate}
            onChange={(e) => setFoodDate(e.target.value)}
          />
        </div>
      </div>

      {/* Symptom Analysis Section */}
      <div className="bg-white shadow-lg rounded-2xl px-6 py-6 w-full max-w-2xl">
        <h2 className="text-2xl font-bold text-blue-800 mb-4">Symptom Analysis</h2>
        <p className="font-bold text-blue-800 mb-4">
          Please choose the symptom(s) you have experienced and rate its/their severity from 1 to 10
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-bold text-blue-700">
          {symptomsList.map((symptom, index) => (
            <div
              key={index}
              className="flex items-center justify-between border p-3 rounded-lg shadow-sm"
            >
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  className="w-3 h-3 border-2 border-blue-700 rounded appearance-none checked:bg-blue-700 checked:border-transparent focus:outline-none"
                  onChange={() => handleSymptomToggle(symptom)}
                  checked={selectedSymptoms[symptom] !== undefined}
                />
                {symptom}
              </label>

              {selectedSymptoms[symptom] !== undefined && (
                <input
                  type="number"
                  min="1"
                  max="10"
                  value={selectedSymptoms[symptom]}
                  onChange={(e) =>
                    handleSeverityChange(symptom, Number(e.target.value))
                  }
                  className="w-16 border p-1 rounded text-center"
                />
              )}
            </div>
          ))}
        </div>

        <input
          type="datetime-local"
          className="mt-4 border p-3 rounded-lg shadow-sm text-blue-700 w-76"
          value={symptomDate}
          onChange={(e) => setSymptomDate(e.target.value)}
        />
      </div>

      {/* Submit Button */}
      <button
        onClick={handleSubmit}
        className="mt-8 bg-blue-700 hover:bg-blue-800 text-white px-6 py-3 rounded-lg font-bold"
      >
        Submit
      </button>

      {/* Logs Table */}
      {savedLogs.length > 0 && (
        <div className="mt-10 w-full max-w-4xl overflow-auto">
          <h3 className="text-xl font-bold mb-4 text-blue-800">Your Logs</h3>
          <table className="min-w-full border border-gray-300 rounded">
            <thead className="bg-blue-100">
              <tr>
                <th className="border px-4 py-2 text-left">Food</th>
                <th className="border px-4 py-2 text-left">Consumed At</th>
                <th className="border px-4 py-2 text-left">Symptom</th>
                <th className="border px-4 py-2 text-left">Severity</th>
                <th className="border px-4 py-2 text-left">Occurred At</th>
              </tr>
            </thead>
            <tbody>
              {savedLogs.map((log) =>
                log.dishes.map((dish, i) =>
                  log.symptoms.map((symptom, j) => (
                    <tr key={`${log._id}-${i}-${j}`}>
                      <td className="border px-4 py-2">{dish.name}</td>
                      <td className="border px-4 py-2">
                        {new Date(dish.consumedAt).toLocaleString()}
                      </td>
                      <td className="border px-4 py-2">{symptom.symptom}</td>
                      <td className="border px-4 py-2">{symptom.severity}</td>
                      <td className="border px-4 py-2">
                        {new Date(symptom.occurredAt).toLocaleString()}
                      </td>
                    </tr>
                  ))
                )
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div className="mb-10 mt-10 bg-blue-100 shadow-lg rounded-2xl px-2 py-2 max-w-50 w-full mx-4 text-center font-bold text-blue-900">
        <p>Made by Stridez</p>
      </div>
    </div>
  );
};

export default UserInput;
