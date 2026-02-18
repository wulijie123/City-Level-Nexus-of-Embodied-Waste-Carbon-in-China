import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from copy import deepcopy
import joblib
import matplotlib.pyplot as plt

# =========================
# 1. Read data
# =========================
data_path = "./data/2015_data.csv"
df = pd.read_csv(data_path).dropna()

# Expected columns:
# city, province, sector20, HW, total_output, GDP, pop, growth, secondaryrate, tertiaryrate, sw

# =========================
# 2. Feature / target definition
# =========================
categorical_features = ["sector20", "province"]

numerical_features = [
    "HW",                
    "total_output",
    "GDP",
    "pop",
    "growth",
    "secondaryrate",
    "tertiaryrate"
]

target = "sw"

required_cols = categorical_features + numerical_features + [target]
df = df.loc[:, required_cols].dropna()

X = df[categorical_features + numerical_features]
y = df[target]

# =========================
# 3. Preprocessing pipeline
# =========================
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# =========================
# 4. Model definition (fixed seed)
# =========================
FIXED_SEED = 2765

rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=11,
    min_samples_leaf=2,
    max_features=0.8,
    n_jobs=-1,
    random_state=FIXED_SEED
)

model = Pipeline([
    ("preprocess", preprocess),
    ("model", rf)
])

# =========================
# 5. Train / test split and training
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=FIXED_SEED
)

sample_weight_train = np.sqrt(y_train)

model.fit(X_train, y_train, model__sample_weight=sample_weight_train)

# =========================
# 6. Model evaluation
# =========================
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
N = len(y_test)

print("====== Model Evaluation ======")
print(f"Seed used            : {FIXED_SEED}")
print(f"R²                   : {r2:.4f}")
print(f"MAE                  : {mae:.2f}")
print(f"RMSE                 : {rmse:.2f}")
print(f"Test sample count N  : {N}")

# =========================
# 7. Save trained pipeline
# =========================
model_save_path = "random_forest_pipeline_seed.joblib"
joblib.dump(deepcopy(model), model_save_path)
print(f"Trained model saved to: {model_save_path}")

# =========================
# 8. Feature importance (aggregated)
# =========================
preprocess_fitted = model.named_steps["preprocess"]
ohe = preprocess_fitted.named_transformers_["cat"]
cat_names = ohe.get_feature_names_out(categorical_features)
feature_names = np.concatenate([cat_names, numerical_features])

importances = model.named_steps["model"].feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

def base_feature_name(col_name):
    if col_name.startswith("sector20_"):
        return "sector20"
    if col_name.startswith("province_"):
        return "province"
    return col_name

importance_df["base_feature"] = importance_df["feature"].apply(base_feature_name)

grouped_importance = (
    importance_df.groupby("base_feature", as_index=False)["importance"]
    .sum()
    .sort_values("importance", ascending=False)
)

print("\n====== Feature Importance (Aggregated) ======")
print(grouped_importance.to_string(index=False))
