import dotenv from "dotenv";
dotenv.config();
import express from "express";
import { OpenAI } from "openai";

const router = express.Router();

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

router.post("/analyze", async (req, res) => {
  const { dishes, symptoms } = req.body;

  const userPrompt = `
A user with possible Crohn's disease consumed the following:
${dishes.map((d) => `${d.name} at ${d.consumedAt}`).join(", ")}.

They experienced the following symptoms:
${symptoms.map((s) => `- ${s.symptom} (Severity: ${s.severity}) at ${s.occurredAt}`).join("\n")}

As a nutrition and health assistant with knowledge of Crohn's disease, please:
1. Identify the likely ingredients in each dish that may trigger or aggravate Crohn's symptoms (e.g., dairy, high-fiber vegetables, gluten, spices, fried foods), and capitalize whatever words starting first in a list line.
2. Based on the symptoms and their severity, advise the user if they should mitigate (e.g., reduce intake), discard (avoid entirely), or adjust (e.g., prepare differently or substitute ingredients) their intake.
3. Keep Crohn's-specific dietary sensitivities in mind (e.g., inflammation triggers, digestibility, hydration needs).

Respond in this strict JSON format:

{
  "ingredients": ["ingredient1", "ingredient2", ...],
  "advice": "Your advice text here."
}
`;

  try {
    const completion = await openai.chat.completions.create({
      model: "gpt-4",
      messages: [{ role: "user", content: userPrompt }],
      temperature: 0.7,
    });

    const message = completion.choices[0].message.content;

    // Try-catch in case the model output isn't well-formed JSON
    try {
      const parsed = JSON.parse(message);
      res.json(parsed);
    } catch (parseErr) {
      console.error("Failed to parse JSON:", parseErr);
      res.status(500).json({
        error: "Model response was not valid JSON.",
        raw: message,
      });
    }
  } catch (err) {
    console.error("OpenAI error:", err);
    res.status(500).json({ error: "Failed to get analysis from GPT." });
  }
});

export default router;
