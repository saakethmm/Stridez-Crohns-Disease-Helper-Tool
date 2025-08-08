Introduction: Model Overview
Our goal was to create a machine learning model that accurately correlates users’ symptoms to specific food ingredients and nutrients, enabling personalized risk assessments of meals. The challenge was addressing the cold start problem — how to offer meaningful insights when user data is initially sparse. To solve this, we designed a hybrid modeling approach:
Random Forest models provide immediate, interpretable predictions during the early “demo” phase, helping users get value from the app right away.


Once enough data accumulates, a deep learning neural network takes over for nuanced, multi-output risk classification and ingredient-level breakdowns.


This approach allows us to deliver continuous value, improving as more user data flows in.

Feature Breakdown: Inputs & Outputs
Inputs (Features):
Ingredients: Ingredient lists for meals logged by users.


Symptoms: User-reported symptoms, including timing relative to meal consumption.


Time Eaten: Timestamp of when the meal was consumed, allowing us to calculate delay until symptom onset.


Meal/Dish Name: Identifier for each logged meal.


Outputs:
Risk Factor (Numeric, 1-10 scale): Quantifies the likelihood that an ingredient or meal triggers symptoms (10 = highest risk, 5 = neutral, 1 = safe).


Meal Classification (Categorical): Overall meal risk level — High, Moderate, or Low risk — derived from the risk factors of constituent ingredients.


Ingredient Breakdown: Risk factors are assigned to individual ingredients within meals, supporting detailed visualization and user insight.


This multi-output setup supports both a high-level summary for users and deep-dive analysis at the ingredient level.
Training Steps & Justifications
Data Preparation:
 We merge baseline nutrient datasets with user food logs and symptom logs, applying hard-coded overrides for known safe and trigger foods. We calculate time delays between eating and symptoms to better capture causality.


Data Encoding:
 Inputs are encoded to prioritize nutrient information, reflecting the hypothesis that certain nutrient classes correlate strongly with symptoms.


Model Choice & Architecture:


We begin with a Random Forest model during the demo phase for immediate, interpretable predictions with limited data.


Once sufficient data is available, we use a multi-output neural network with two outputs (risk factor and classification), optimized via hyperparameter tuning (units, dropout rate, learning rate).


Dropout and other techniques help avoid overfitting.


Output Strategy:
 The model simultaneously predicts the risk factor for individual ingredients and classifies the overall meal, supporting both user-facing visualizations and backend risk scoring.


Risk Factor Scale:
 Using a 1-10 scale allows nuanced user feedback incorporation (e.g., severity multipliers for symptoms like vomiting vs. bloating) and easy visualization (red/yellow/green coding).


Hyperparameter Tuning:
 We implemented a Keras Tuner grid search to identify the optimal model configuration, improving accuracy and generalizability.


Future Iterations: Task Breakdown with Model Development 
To complement the numerical risk scoring, we plan to integrate an LLM (ChatGPT4.0) to analyze the merged dataset and generate personalized meal recommendations. This has several benefits:
The LLM can interpret complex symptom severity and time correlations better than fixed numeric scores.


It helps generate natural language meal plans tailored to the user's dietary restrictions and symptom history.


This integration offers an adaptive, conversational nutrition assistant experience.


The LLM consumes the merged datasets combined with recipe lists found from trusted dietary sites, and the symptom severity scoring we hope to work on in the future as well to propose gut-friendly meals, enhancing user engagement and trust.
