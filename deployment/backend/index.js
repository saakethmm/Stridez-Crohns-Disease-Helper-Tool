import express from 'express';
import { mongoDBURL, PORT } from './config.js';
import mongoose from 'mongoose';
import { FoodLog } from './models/foodModel.js';
import cors from 'cors';
import gptRoute from './gpt.js'

const app = express();

//middleware to parse JSON request bodies
app.use(express.json());
app.use(cors());
app.use("/gpt", gptRoute);

//route to save new user input
app.post('/user/input', async (req, res) => {
  try {
    const { userId, dishes, symptoms, submittedAt } = req.body;

    //validation
    if (!dishes?.length || !symptoms?.length) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const newFoodLog = new FoodLog({
      userId,
      dishes,
      symptoms,
      submittedAt: submittedAt || new Date(),
    });

    const savedLog = await newFoodLog.save();
    res.status(201).json(savedLog);
  } catch (error) {
    console.error("Error saving food log:", error);
    res.status(500).json({ error: 'Server error while saving user input' });
  }
});

//route to get new user input
app.get('/user/input/:id', async (req, res) => {
  try {
    const { id } = req.params;

    // Validate that id is a valid ObjectId
    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({ error: 'Invalid id format' });
    }

    const foodLog = await FoodLog.findById(id);
    if (!foodLog) {
      return res.status(404).json({ error: 'Food log not found' });
    }

    res.status(200).json(foodLog);
  } catch (error) {
    console.error(error.message);
    res.status(500).json({ error: 'Server error while fetching food log' });
  }
});

//connect to MongoDB and start the server
mongoose
  .connect(mongoDBURL)
  .then(() => {
    console.log('Connected to MongoDB');
    app.listen(PORT, () => {
      console.log('Server running on http://localhost:3000');
    });
  })
  .catch((error) => {
    console.log('MongoDB connection error:', error);
  });
