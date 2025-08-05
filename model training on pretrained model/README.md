# NutrifyMe-A-Personalized-Diet-Recommendation-System-Using-Deep-Learning

### Overview
With the emergence of personalized health solutions,
nutrition and dietary recommendations have made tremendous
strides. This project proposes the establishment of a Personalized
diet recommendation system that leverages deep learning to
produce diet plans that address individual nutritional needs.
Using a robust dataset with rich nutritional data, the model
predicts calorie and macronutrient components of several food
items. Given the user input (calories, protein, fat, and carbohydrate specifications), the system will recommend food items
that match the user’s nutritional preferences. This personalized
meal-planning approach encourages users to engage in healthy
eating behaviors by providing individualized meal suggestions in
real-time. The deep learning model was constructed and trained
utilizing Keras and TensorFlow for food selection based upon
users’ nutritional allocation. After evaluating the food items
based on the nutritional gaps, we recommend the top 10 food
based on personal health parameters.



---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technologies](#technologies)
4. [Installation](#installation)
5. [Dataset](#dataset)
6. [Model Architecture](#model-architecture)
7. [Results](#results)
8. [Future Enhancements](#future-enhancements)


---

## Project Overview
In response to the growing interest in personalized nutrition, NutrifyMe provides tailored dietary advice using deep learning techniques. Traditional dietary recommendations lack the flexibility to address individual needs. NutrifyMe bridges this gap by analyzing user data to provide customized, actionable meal plans that adapt over time.

## Features
- **Personalized Recommendations:** Generates daily meal plans based on user-defined macronutrient goals (calories, protein, fat, and carbs).
- **Nutrition Analysis:** Predicts nutritional content for diverse food items.
- **Real-Time Feedback:** Updates suggestions dynamically based on user preferences and health data.
- **Top 10 Food Recommendations:** Recommends food items ranked by nutritional suitability.

---

## Technologies
- **Frameworks:** TensorFlow, Keras
- **Programming Language:** Python
- **Libraries:** Numpy, Pandas, Scikit-learn

---

## Dataset

The dataset used for **NutrifyMe** includes detailed nutritional information for various food items, enabling tailored dietary recommendations based on caloric and macronutrient needs. This dataset consists of 722 entries and 37 columns, providing a comprehensive breakdown of essential dietary components, including:

- **Caloric Value**: Total calorie count per serving.
  
- **Macronutrients**:
  - **Fat**: Includes total fat, saturated fats, monounsaturated fats, and polyunsaturated fats.
  - **Carbohydrates**: Total carbohydrates, sugars, and dietary fiber.
  - **Protein**: Protein content per serving.

- **Micronutrients**:
  - **Vitamins**: A range of vitamins, including A, B1, B12, C, D, E, and K.
  - **Minerals**: Includes essential minerals like calcium, iron, magnesium, potassium, selenium, and zinc.

### Key Dataset Statistics
- **Average Caloric Value**: ~124 calories per item.

- **Macronutrients**:
  - Average **fat content**: ~5.6g, ranging from 0g to 218g.
  - Average **carbohydrates**: ~15.8g, with sugars contributing an average of 3.1g.
  - Average **protein content**: ~3.5g.

- **Micronutrients**:
  - **Calcium**: Mean of 37.7 mg, with a maximum of 868 mg.
  - **Potassium**: Mean of 200.9 mg, with values reaching up to 4053 mg.

Each entry also includes a **Nutrition Density** score to assess the nutrient richness of foods, assisting the model in ranking recommendations based on user health goals.

---

## Model Architecture

The **NutrifyMe** model is a neural network built with TensorFlow and Keras to generate personalized dietary recommendations based on user-defined nutritional goals. The architecture and processing pipeline are as follows:

1. **Data Preprocessing**:
   - **Data Cleaning**: Removes unnecessary columns and fills missing values with zeros to maintain consistency.
   - **Normalization**: Standard scaling is applied to ensure features are on a consistent scale, optimizing the model’s performance.

2. **Model Architecture**:
   - **Input Layer**: Receives user inputs related to caloric and macronutrient needs.
   - **Hidden Layers**:
     - **Dense Layers**: Multiple fully connected dense layers, each with ReLU (Rectified Linear Unit) activation, capture complex relationships in the data.
     - **Dropout Layers**: Dropout is used between dense layers to prevent overfitting by randomly dropping neurons during training.
   - **Output Layer**: A single dense layer with linear activation predicts the Nutrition Density score for each food item, indicating how closely it matches the user’s requirements.

3. **Compilation and Training**:
   - **Optimizer**: Adam optimizer is used for efficient and adaptive learning.
   - **Loss Function**: Mean Squared Error (MSE) is selected for error minimization between predicted and actual nutrition values.
   - **Training Parameters**: The model is trained over multiple epochs to ensure convergence, with batch size set to optimize computation time and resource efficiency.

4. **Evaluation Metrics**:
   - **Mean Absolute Error (MAE)** and **R-squared (R²)** are calculated on the test set to evaluate the model’s predictive accuracy and performance.

This architecture allows **NutrifyMe** to make accurate, personalized food recommendations, ranking food items based on their nutritional suitability to meet user-specific dietary goals.

---

## Results
NutrifyMe effectively tailors diet plans by predicting nutrient content based on user-defined goals. It ranks food items, recommending the top 10 options that best match individual health parameters. User satisfaction surveys indicate high relevance of recommendations to daily dietary needs.

---

## Future Enhancements
Potential improvements include:
- **Real-Time Data Integration:** Connecting with wearable devices for dynamic dietary recommendations.
- **Expanded Data Sources:** Incorporating genetic, lifestyle, and cultural preferences.
- **Reinforcement Learning:** Adding real-time adaptation based on user feedback.

---

