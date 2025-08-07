import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch #imports

#load paths from before
MERGED_DATA_PATH = "data\merged_dataset.csv"
MODEL_SAVE_PATH = "models/model.h5" #where to save model for retrieval
MAX_TRIALS = 5 # number of diff params sets to try
EPOCHS = 30
BATCH_SIZE = 32

#loading data first
if not os.path.exists(MERGED_DATA_PATH):
    raise FileNotFoundError(f"{MERGED_DATA_PATH} not found. Run prepareModel.py first.")

df = pd.read_csv(MERGED_DATA_PATH)
print(f"Loaded Merged Dataset: {df.shape}")

#define the inputs and outputs
feature_cols = [col for col in df.columns if col not in ["risk factor", "classification"]]
X = df[feature_cols].values
y_risk = df["risk factor"].values

#encode the classification models
if df["classification"].dtype == object:
    le = LabelEncoder()
    y_class = le.fit_transform(df["classification"])
else:
    y_class = df["classification"].values

#split data
X_train, X_test, y_risk_train, y_risk_test, y_class_train, y_class_test = train_test_split(
    X, y_risk, y_class, test_size=0.2, random_state=42
)

# scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# the hypermodel class
class RiskHyperModel(HyperModel):
    def build(self, hp):
        input_layer = Input(shape=(X_train_scaled.shape[1],), name="input_layer")

        # shared layers
        x = Dense(
            units=hp.Int('units1', min_value=32, max_value=128, step=32),
            activation='relu'
        )(input_layer)
        x = Dropout(hp.Float('dropout1', 0.1, 0.5, step=0.1))(x)
        x = Dense(
            units=hp.Int('units2', min_value=16, max_value=64, step=16),
            activation='relu'
        )(x)

        # risk factor output
        risk_output = Dense(1, activation="linear", name="risk_output")(x)

        # classification output
        class_output = Dense(3, activation="softmax", name="class_output")(x)

        model = Model(inputs=input_layer, outputs=[risk_output, class_output])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            ),
            loss={
                "risk_output": "mse",
                "class_output": "sparse_categorical_crossentropy"
            },
            metrics={
                "risk_output": ["mae"],
                "class_output": ["accuracy"]
            }
        )
        return model

# rnning the tuner
hypermodel = RiskHyperModel()

tuner = RandomSearch(
    hypermodel,
    objective='val_loss',
    max_trials=MAX_TRIALS,
    executions_per_trial=1,
    directory='tuning',
    project_name='risk_factor_tuning'
)

tuner.search(
    X_train_scaled,
    {"risk_output": y_risk_train, "class_output": y_class_train},
    validation_data=(X_test_scaled, {"risk_output": y_risk_test, "class_output": y_class_test}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# gets the best model and saves it 
best_model = tuner.get_best_models(num_models=1)[0]
best_model.save(MODEL_SAVE_PATH)
print(f"Best model saved to {MODEL_SAVE_PATH}")