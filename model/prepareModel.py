import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

#initial config of datafiles
BASE_FILE_PATH = "data\base_dataset.csv"
USER_FOOD_PATH = "data\food_log.csv"
USER_SYMPTOMS_PATH = "data\symptom_log.csv" 
USER_SAFE_TRIGGER_PATH = "data\safe_trigger_foods.csv"
OUTPUT_PATH = "data\merged_dataset.csv" #model should automatically create this file and store it in the data folder

DEFAULT_RISK = 5 #neutral starting risk

#create function to load user entered safeties & triggers
def load_user_overrides():
    if not os.path.exists(USER_SAFE_TRIGGER_PATH):
        print("No CSV found for overriding. Will continue with default risk factors for all foods.")
        return [], []
    overrides_df = pd.read_csv(USER_SAFE_TRIGGER_PATH)

    # assigning safe foods first
    safe_foods = overrides_df.loc[overrides_df["type"].str.lower() == "safe", "food"].tolist()
    # assigning trigger foods
    trigger_foods = overrides_df.loc[overrides_df["type"].str.lower() == "trigger", "food"].tolist()
    
    return safe_foods, trigger_foods

#func to applythose user overrides to default dataset
def apply_user_overrides(df, safe_list, trigger_list, risk_col="risk_score"):
    df[risk_col] = DEFAULT_RISK  # starting with neutral risk factor

    # sorting safeties with risk value of 1
    for food in safe_list:
        df.loc[df["ingredients"].str.contains(food, case=False, na=False), risk_col] = 1
    # sorting triggers with risk value of 10
    for food in trigger_list:
        df.loc[df["ingredients"].str.contains(food, case=False, na=False), risk_col] = 10

    return df

    
#func for calculating the time delay between logged foods and symptoms to help with correlation/predictions
def calculate_time_delay(row, eaten_col="time_eaten", symptom_col="symptom_start_time"): # i think these columns should also match how the user input is being saved/appended in the csv files which we'll figure out after having it tested
    try:
        eaten_time = pd.to_datetime(row[eaten_col], format="%H:%M")
        symptom_time = pd.to_datetime(row[symptom_col], format="%H:%M")
        return (symptom_time - eaten_time).total_seconds() / 60
    except Exception:
        return np.nan

# applying one hot encoding to the columns
def one_hot_encode(df, column_name): #could potentially be switched to embedding i think?? not too knowledgeable on that
    encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))
    return pd.concat([df.reset_index(drop=True), encoded_df], axis=1)

# onto main func
def main():
    #load up dataset and print confirmation 
    if os.path.exists(BASE_FILE_PATH):
        base_df = pd.read_csv(BASE_FILE_PATH)
        print(f"Loaded base dataset: {base_df.shape}")
    else:
        base_df = pd.DataFrame()
    #load in user food and symptom logs
    food_df = pd.read_csv(USER_FOOD_PATH)
    symptom_df = pd.read_csv(USER_SYMPTOMS_PATH)

    # merge base nutrient data into food log
    food_df = pd.merge(food_df, base_df, on="ingredients", how="left")

    # load safe foods and triggers
    safe_foods, trigger_foods = load_user_overrides()

    # apply the risk factor overrides (from user defaults)
    food_df = apply_user_overrides(food_df, safe_foods, trigger_foods)

    # time-based merge with symptom log
    merged_df = pd.merge_asof(
        food_df.sort_values("time_eaten"),
        symptom_df.sort_values("symptom_start_time"),
        left_on="time_eaten",
        right_on="symptom_start_time",
        direction="forward",  # change direction if needed
        tolerance=pd.Timedelta("6h")  # only match symptoms within 6 hours
    )

    # Capture the meal/ingredients linked to each symptom
    if "ingredients" in merged_df.columns:
        merged_df["recent_food"] = merged_df["ingredients"]
    elif "Meal" in merged_df.columns:  # if you have a Meal column instead
        merged_df["recent_food"] = merged_df["Meal"]
    else:
        merged_df["recent_food"] = "N/A"

    # calculate time delay for later use
    merged_df["time_delay_minutes"] = merged_df.apply(calculate_time_delay, axis=1)

    # keep categorical info (risk factor + time delay + severity for training)
    numeric_cols = merged_df.select_dtypes(include=[np.number]).columns.tolist()
    important_cols = numeric_cols + ["risk_score", "ingredients", "symptoms", "time_eaten", "symptom_start_time"]
    merged_df = merged_df[[col for col in important_cols if col in merged_df.columns]]

    # save processed dataset for further use!
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset successfully saved to {OUTPUT_PATH}")

    # save processed dataset for further use!
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Merged dataset successfully saved to {OUTPUT_PATH}")

    # also save separate symptom correlation file for LLM 
    correlation_cols = ["time_eaten", "ingredients", "symptom_start_time", "symptoms", "time_delay_minutes"]
    symptom_corr_df = merged_df[[col for col in correlation_cols if col in merged_df.columns]].dropna(subset=["symptoms"])
    symptom_corr_df.to_csv("data/symptom_correlations.csv", index=False)
    print("Symptom correlations saved to data/symptom_correlations.csv")
