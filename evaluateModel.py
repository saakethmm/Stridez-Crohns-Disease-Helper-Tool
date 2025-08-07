
# evaluateModel.py IMPROVED VERSION


























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
