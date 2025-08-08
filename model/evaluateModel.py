
# evaluateModel.py 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import joblib

# =======================
# 1. Load Data & Model
# =======================
df = pd.read_csv(r"..\deployment\data\merged_dataset.csv")

# Select numeric feature columns exactly like during training
feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Remove your target columns if they are numeric and present
for col in ["risk_score", "risk factor", "classification"]:
    if col in feature_cols:
        feature_cols.remove(col)

# Prepare X for prediction
X = df[feature_cols]

# Load scaler used during training and apply it here
scaler = joblib.load("models/scaler.pkl")  # Make sure you saved this during training!

X_scaled = scaler.transform(X)

# Load trained model
rf_model = joblib.load(r"models/rf_demo_model.pkl")

ingredients_col = df["Ingredients"] if "Ingredients" in df.columns else None
nutrients_col = df["Nutrients"] if "Nutrients" in df.columns else None


# Predict risk
predicted_risk = rf_model.predict(X_scaled)

df["PredictedRisk"] = predicted_risk

# Classification thresholds
def classify_risk(score):
    if score > 5:
        return "High Risk"
    elif score >= 3:
        return "Moderate Risk"
    else:
        return "Low Risk"

df["RiskClass"] = df["PredictedRisk"].apply(classify_risk)

# =======================
# 3. Sorted Description Risk Table
# =======================
meal_risk_df = df[["Description", "PredictedRisk", "RiskClass"]].drop_duplicates()
meal_risk_df = meal_risk_df.sort_values(by="PredictedRisk", ascending=False)

print("\n===  Risk Table (Sorted) ===")
print(meal_risk_df)

# =======================
# 4. Ingredient Risk Pie Chart
# =======================
if ingredients_col is not None:
    all_ingredients = []
    for meal, risk in zip(ingredients_col, df["PredictedRisk"]):
        for ing in str(meal).split(","):
            all_ingredients.append((ing.strip(), risk))
    ing_df = pd.DataFrame(all_ingredients, columns=["Ingredient", "Risk"])
    ing_risk = ing_df.groupby("Ingredient")["Risk"].mean().sort_values(ascending=False).head(10)

    colors = ing_risk.apply(lambda r: "red" if r > 5 else "yellow" if r >= 3 else "green")
    plt.figure(figsize=(6,6))
    plt.pie(ing_risk, labels=ing_risk.index, autopct="%1.1f%%", colors=colors)
    plt.title("Top 10 Risky Ingredients")
    plt.tight_layout()
    plt.show()

# =======================
# 5. Nutrient Risk Pie Chart & Correlation
# =======================
if nutrients_col is not None:
    nutrient_data = []
    for nuts, risk, symptoms in zip(nutrients_col, df["PredictedRisk"], df.get("Symptoms", [""]*len(df))):
        for n in str(nuts).split(","):
            nutrient_data.append((n.strip(), risk, 1 if str(symptoms).strip() else 0))
    nut_df = pd.DataFrame(nutrient_data, columns=["Nutrient", "Risk", "SymptomTriggered"])

    nut_risk = nut_df.groupby("Nutrient")["Risk"].mean().sort_values(ascending=False).head(10)
    colors = nut_risk.apply(lambda r: "red" if r > 5 else "yellow" if r >= 3 else "green")

    # Pie chart
    plt.figure(figsize=(6,6))
    plt.pie(nut_risk, labels=nut_risk.index, autopct="%1.1f%%", colors=colors)
    plt.title("Top 10 Risky Nutrients")
    plt.tight_layout()
    plt.show()

    # Correlation matrix
    pivot_df = nut_df.pivot_table(index="Nutrient", values="SymptomTriggered", aggfunc=np.mean)
    corr_matrix = pivot_df.corr()
    plt.figure(figsize=(5,4))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Nutrient Symptom Correlation")
    plt.tight_layout()
    plt.show()

# =======================
# 6. Save Final CSV
# =======================
df.to_csv("final_predictions.csv", index=False)
print("\n Final CSV saved as final_predictions.csv")
#Import mealGenerator
# from mealGenerator import generate_meal_recommendations

# # After  evaluation and saving final CSV with predictions:
# recommendations = generate_meal_recommendations("merged_dataset_with_symptoms.csv", "gut_friendly_recipes.csv")
# print("\n===== Meal Recommendations =====")
# print(recommendations)

