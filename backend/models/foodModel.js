import mongoose from "mongoose";

// Dish schema: represents each food item consumed
const dishSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true,
  },
  consumedAt: {
    type: Date,
    required: true,
  },
});

// Symptom schema: represents symptoms experienced and their severity
const symptomSchema = new mongoose.Schema({
  symptom: {
    type: String,
    required: true,
  },
  severity: {
    type: Number,
    required: true,
    min: 1,
    max: 10,
  },
  occurredAt: {
    type: Date,
    required: true,
  },
});

// FoodLog schema: stores overall user food + symptom log
const foodLogSchema = new mongoose.Schema({
  dishes: [dishSchema],        // Array of consumed dishes
  symptoms: [symptomSchema],   // Array of symptoms with severity and timestamps
  submittedAt: {
    type: Date,
    default: Date.now,
  },
});

export const FoodLog = mongoose.model("Food", foodLogSchema);
