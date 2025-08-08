
# evaluateModel.py IMPROVED VERSION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

# =======================
# 1. Load Data & Model
# =======================
model = load_model("model.h5")
df = pd.read_csv("merged_dataset.csv")

# Assumming  already preprocessed/encoded during training
X = df.drop(columns=["Meal", "Risk_score", "Symptoms", "Ingredients", "Nutrients"], errors="ignore")
meals = df["Meal"]
ingredients_col = df["Ingredients"] if "Ingredients" in df.columns else None
nutrients_col = df["Nutrients"] if "Nutrients" in df.columns else None

# =======================
# 2. Predict Risk Factors
# =======================
predicted_risk = model.predict(X).flatten()
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
# 3. Sorted Meal Risk Table
# =======================
meal_risk_df = df[["Meal", "PredictedRisk", "RiskClass"]].drop_duplicates()
meal_risk_df = meal_risk_df.sort_values(by="PredictedRisk", ascending=False)

print("\n=== Meal Risk Table (Sorted) ===")
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
from mealGenerator import generate_meal_recommendations

# After  evaluation and saving final CSV with predictions:
recommendations = generate_meal_recommendations("merged_dataset_with_symptoms.csv", "gut_friendly_recipes.csv")
print("\n===== Meal Recommendations =====")
print(recommendations)


























# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from tensorflow.keras.models import load_model
# from sklearn.metrics import confusion_matrix
# import joblib

# # ----------------- STEP 1: Load Model & Preprocessors -----------------
# model = load_model("model.h5")
# tfidf = joblib.load("tfidf_ingredients.pkl")
# time_encoder = joblib.load("time_encoder.pkl")

# # ----------------- STEP 2: Load Evaluation Data -----------------
# data = pd.read_csv("merged_dataset.csv")

# # Optional: Drop any NaNs if needed
# data = data.dropna(subset=["ingredients", "time_eaten", "label"])

# # ----------------- STEP 3: Prepare Features -----------------
# # TF-IDF for ingredients
# ingredient_features = tfidf.transform(data["ingredients"]).toarray()

# # One-hot encode time_eaten
# time_features = time_encoder.transform(data[["time_eaten"]]).toarray()

# # Combine all features
# X_eval = np.concatenate([ingredient_features, time_features], axis=1)

# # Prepare labels (one-hot encoded)
# y_eval = pd.get_dummies(data["label"]).values

# # ----------------- STEP 4: Evaluate Model -----------------
# test_loss, test_acc = model.evaluate(X_eval, y_eval, verbose=0)
# print(f"Evaluation Accuracy: {test_acc:.4f}")

# # ----------------- STEP 5: Predict -----------------
# y_pred_probs = model.predict(X_eval)
# y_pred_classes = np.argmax(y_pred_probs, axis=1)
# y_true_classes = np.argmax(y_eval, axis=1)


# # ----------------- STEP 6: Add Predictions to CSV -----------------
# data["risk_score"] = y_pred_probs[1, 10]   # Probability of Unsafe
# data["classification"] = y_pred_classes  #  1 = Safe, 10 = Unsafe

# # Save final predictions
# data.to_csv("final_predictions.csv", index=False)

# # ----------------- STEP 7: Ingredient Risk Bar Chart -----------------
# from collections import defaultdict

# ingredient_risks = defaultdict(list)

# # Loop over each row and assign risk score to each ingredient
# for i, row in data.iterrows():
#     ingredients = row["ingredients"].split(",")
#     score = row["risk_score"]
#     for ing in ingredients:
#         ing = ing.strip().lower()
#         ingredient_risks[ing].append(score)
        
# # Average risk per ingredient
# avg_risks = {ing: np.mean(scores) for ing, scores in ingredient_risks.items()}

# # Sort and plot
# sorted_risks = dict(sorted(avg_risks.items(), key=lambda x: x[1], reverse=True))

# plt.figure(figsize=(10, 6))
# sns.barplot(x=list(sorted_risks.values()), y=list(sorted_risks.keys()))
# plt.xlabel("Average Risk Score")
# plt.ylabel("Ingredient")
# plt.title("Ingredient Risk Scores")
# plt.tight_layout()
# plt.savefig("ingredient_risk_chart.png")
# plt.show()

# # ----------------- STEP 9: Ingredient-Level Classification -----------------
# ingredient_classification = {
#     ing: 'Unsafe' if score > 0.5 else 'Safe'
#     for ing, score in avg_risks.items()
# }

# ingredient_df = pd.DataFrame({
#     'ingredient': list(ingredient_classification.keys()),
#     'avg_risk_score': list(avg_risks.values()),
#     'classification': list(ingredient_classification.values())
# })
# ingredient_df.to_csv("ingredient_classifications.csv", index=False)

# print("\nAll evaluation results saved!")
