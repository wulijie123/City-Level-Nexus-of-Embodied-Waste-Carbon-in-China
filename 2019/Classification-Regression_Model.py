import os
import joblib
import warnings
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
)
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =========================
# CONFIGURATION
# =========================
data_path = "./data/2015_data_all.csv"
regressor_path = "random_forest_pipeline_seed.joblib"  # regression pipeline (must exist)
seed = 93
classifier_random_state = 0  # used to initialize classifier if needed

# New-data file used for out-of-sample prediction
new_data_path = "./result/inventory/2019HW_all.csv"
out_new_csv = "./result/inventory/2019_newdata_combined_predictions.csv"
out_deleted_csv = "2019_newdata_deleted_rows.csv"

# =========================
# 1. Read training/labeling data
# =========================
print("Loading dataset:", data_path)
df = pd.read_csv(data_path).copy()

# Expected columns for training/classification:
# sector20, HW, sw, GDP, pop, growth, secondaryrate, tertiaryrate, total_output, city, province

# =========================
# 2. Feature and target definitions (use exact supplied column names)
# =========================
categorical_features = ["sector20", "province"]
numerical_features = [
    "HW", "GDP", "pop", "growth",
    "secondaryrate", "tertiaryrate", "total_output"
]
original_target = "sw"

required_cols = categorical_features + numerical_features + [original_target, "city"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise RuntimeError(f"Required columns missing from training file: {missing}")

# Keep only rows where the required columns are present and non-null (robust)
df = df.loc[:, required_cols + ["city"]].copy()
df = df.dropna(subset=required_cols)

# continuous ground truth (used later for regression evaluation)
y_continuous_all = pd.to_numeric(df[original_target], errors="coerce").astype(float)
df[original_target] = y_continuous_all
df = df.dropna(subset=[original_target])  # ensure target numeric

# binary label for classification: 1 if sw != 0, else 0
y_binary = (df[original_target] != 0).astype(int)

X = df[categorical_features + numerical_features].copy()

print("Class distribution (1 = sw != 0):")
print(y_binary.value_counts().to_string())
print("Proportions:")
print(y_binary.value_counts(normalize=True).to_string())

# =========================
# 3. Preprocessing pipeline (shared)
# =========================
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ("num", StandardScaler(), numerical_features)
    ]
)

# =========================
# 4. Classification model
# =========================
clf = RandomForestClassifier(
    n_estimators=400,
    max_depth=11,
    min_samples_leaf=2,
    max_features=0.8,
    n_jobs=-1,
    random_state=classifier_random_state
)
model_clf = Pipeline([("preprocess", preprocess), ("model", clf)])

# =========================
# 5. Try seeds for classification and pick best split (ROC AUC preferred)
# =========================
results = []
best_score = -np.inf
best_seed = None
best_metrics = {}
best_split = None

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=seed, stratify=y_binary
    )
except ValueError:
    # fallback if stratify fails
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=seed
    )

model_clf.fit(X_train, y_train)
y_pred = model_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

try:
    y_score = model_clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_score)
except Exception:
    roc_auc = np.nan
    y_score = None

print(f"seed: {seed:4d} | ACC: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {np.nan if np.isnan(roc_auc) else f'{roc_auc:.4f}'}")

results.append({
    "seed": seed,
    "accuracy": acc,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "roc_auc": roc_auc
})

score_for_compare = roc_auc if not np.isnan(roc_auc) else f1
if score_for_compare > best_score:
    best_score = score_for_compare
    best_seed = seed
    best_metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}
    best_split = (deepcopy(X_train), deepcopy(X_test), deepcopy(y_train), deepcopy(y_test))

# =========================
# 6. Summary of classification seeds tried
# =========================
results_df = pd.DataFrame(results)
print("\n====== Summary of classification seeds ======")
if not results_df.empty:
    print(f"Accuracy : {results_df['accuracy'].mean():.4f} ± {results_df['accuracy'].std():.4f}")
    print(f"Precision: {results_df['precision'].mean():.4f} ± {results_df['precision'].std():.4f}")
    print(f"Recall   : {results_df['recall'].mean():.4f} ± {results_df['recall'].std():.4f}")
    print(f"F1       : {results_df['f1'].mean():.4f} ± {results_df['f1'].std():.4f}")
    if results_df['roc_auc'].notna().any():
        print(f"ROC AUC  : {results_df['roc_auc'].dropna().mean():.4f} ± {results_df['roc_auc'].dropna().std():.4f}")
else:
    print("No classification seed results available.")

print("\n====== Best classification split ======")
print(f"Best Seed : {best_seed}")
print(f"Metrics   : {best_metrics}")

if best_seed is None or best_split is None:
    raise RuntimeError("Unable to determine best seed/split. Please check seeds_to_try and data distribution.")

# =========================
# 7. Retrain classifier on best split (reproducible) and prepare regression pipeline
# =========================
X_train_best, X_test_best, y_train_best, y_test_best = best_split
model_clf.fit(X_train_best, y_train_best)  # retrain for reproducibility

# Align continuous test-ground-truth with X_test_best indices
if hasattr(X_test_best, "index"):
    test_idx = X_test_best.index
    y_test_continuous = y_continuous_all.loc[test_idx].values
else:
    # fallback: take first N in original continuous series (unlikely)
    y_test_continuous = y_continuous_all.values[:len(X_test_best)]

# =========================
# 8. Load regression pipeline (must exist)
# =========================
if os.path.exists(regressor_path):
    reg_pipeline = joblib.load(regressor_path)
    print(f"Loaded regression pipeline from: {regressor_path}")
else:
    raise RuntimeError(f"Regression pipeline not found at '{regressor_path}'. Cannot proceed with regression predictions.")

# =========================
# 9. Combined prediction on test split: classifier -> reg (label==1 -> reg, label==0 -> 0)
# =========================
clf_pred_label = model_clf.predict(X_test_best)  # 1 -> non-zero, 0 -> zero
y_pred_quantity = np.zeros(len(X_test_best), dtype=float)

nonzero_mask = (clf_pred_label == 1)
if nonzero_mask.sum() > 0:
    # preserve DataFrame indexing if present
    X_reg_input = X_test_best.loc[nonzero_mask] if isinstance(X_test_best, pd.DataFrame) else X_test_best[nonzero_mask]
    reg_preds = reg_pipeline.predict(X_reg_input)
    y_pred_quantity[nonzero_mask] = np.array(reg_preds).ravel()

# compute overall regression metrics against continuous truth
r2_overall = r2_score(y_test_continuous, y_pred_quantity)
mae_overall = mean_absolute_error(y_test_continuous, y_pred_quantity)
rmse_overall = np.sqrt(mean_squared_error(y_test_continuous, y_pred_quantity))

print("\n====== Combined (classification+regression) results on test split ======")
print(f"Test samples: {len(y_test_continuous)}")
print(f"Classifier predicted non-zero: {int(nonzero_mask.sum())}")
print(f"Classifier predicted zero    : {int((~nonzero_mask).sum())}")
print(f"Overall R² : {r2_overall:.4f}")
print(f"Overall MAE: {mae_overall:.2f}")
print(f"Overall RMSE: {rmse_overall:.2f}")

# Metrics restricted to predicted-nonzero subset (if any)
if nonzero_mask.sum() > 0:
    true_nonzero = y_test_continuous[nonzero_mask]
    pred_nonzero = y_pred_quantity[nonzero_mask]
    r2_nonzero = r2_score(true_nonzero, pred_nonzero) if len(np.unique(true_nonzero)) > 1 else np.nan
    mae_nonzero = mean_absolute_error(true_nonzero, pred_nonzero)
    rmse_nonzero = np.sqrt(mean_squared_error(true_nonzero, pred_nonzero))
    print("\n====== Regression metrics on predicted-nonzero subset ======")
    print(f"N_nonzero : {len(true_nonzero)}")
    print(f"R²_nonzero: {np.nan if np.isnan(r2_nonzero) else f'{r2_nonzero:.4f}'}")
    print(f"MAE_nonzero: {mae_nonzero:.2f}")
    print(f"RMSE_nonzero: {rmse_nonzero:.2f}")
else:
    print("\nNote: Classifier predicted zero for all test samples; no regression subset metrics available.")

# =========================
# 10. Save test-split combined predictions
# =========================
out_df = X_test_best.copy().reset_index(drop=True)
out_df["y_true_sw"] = y_test_continuous
out_df["clf_pred_label_nonzero"] = clf_pred_label
try:
    clf_pred_proba = model_clf.predict_proba(X_test_best)[:, 1]
    out_df["clf_pred_nonzero_proba"] = clf_pred_proba
except Exception:
    pass
out_df["predicted_sw_quantity"] = y_pred_quantity

out_csv = "testset_combined_predictions.csv"
out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\nSaved test-set combined predictions to: {out_csv}")

# =========================
# 11. Plot predicted vs true (density scatter)
# =========================
y_true = np.array(y_test_continuous).ravel()
y_pred = np.array(y_pred_quantity).ravel()
N = len(y_true)

AX_MAX = 4e6
bins = 42

y_true_c = np.clip(y_true, 0, AX_MAX)
y_pred_c = np.clip(y_pred, 0, AX_MAX)

H, xedges, yedges = np.histogram2d(y_true_c, y_pred_c, bins=bins, range=[[0, AX_MAX], [0, AX_MAX]])
x_idx = np.clip(np.digitize(y_true_c, xedges) - 1, 0, H.shape[0] - 1)
y_idx = np.clip(np.digitize(y_pred_c, yedges) - 1, 0, H.shape[1] - 1)
point_counts = H[x_idx, y_idx]

vmin = 1
vmax = np.percentile(point_counts, 99)

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
sc = ax.scatter(y_true, y_pred, c=point_counts, s=40, cmap="jet", alpha=0.9, edgecolors="none")
sc.set_clim(vmin, vmax)

ax.plot([0, AX_MAX], [0, AX_MAX], color="#8B0000", linewidth=4)
coef = np.polyfit(y_true_c, y_pred_c, 1)
x_fit = np.linspace(0, AX_MAX, 300)
ax.plot(x_fit, coef[0] * x_fit + coef[1], color="#4f4f4f", linewidth=4)

ax.set_xlim(0, AX_MAX)
ax.set_ylim(0, AX_MAX)
ax.set_xlabel("Declared generation quantity of SW (t)", fontsize=21)
ax.set_ylabel("Predicted generation quantity of SW (t)", fontsize=21)
ax.ticklabel_format(style="sci", axis="both", scilimits=(6, 6))
ax.tick_params(labelsize=21)
ax.xaxis.get_offset_text().set_fontsize(18)
ax.yaxis.get_offset_text().set_fontsize(18)

ax.text(0.1, 0.95, f"R² = {r2_overall:.2f}\nRMSE = {rmse_overall:.2f}\nN = {N}", transform=ax.transAxes, va="top", fontsize=21)

cax = fig.add_axes([0.78, 0.15, 0.03, 0.3])
cbar = plt.colorbar(sc, cax=cax)
desired_ticks = np.array([100, 300, 500])
valid_ticks = desired_ticks[(desired_ticks >= vmin) & (desired_ticks <= vmax)]
if len(valid_ticks) > 0:
    cbar.set_ticks(valid_ticks)
    cbar.set_ticklabels([str(t) for t in valid_ticks])
else:
    ticks = np.linspace(vmin, vmax, 4, dtype=int)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([str(t) for t in ticks])
cbar.set_label("Count", fontsize=18)
cbar.ax.tick_params(labelsize=18)

plt.tight_layout()
plt.savefig("combined_pred_vs_true_density.png", dpi=800)
plt.show()

# =========================
# 12. New data prediction pipeline (diagnostics + predictions)
# =========================
print("\n====== New-data prediction (diagnostics + predictions) ======")
# 12.1 Load new data
df_new_raw = pd.read_csv(new_data_path)
print("New data original rows:", len(df_new_raw))

# Diagnostic copy and delete-reason column
df_diag = df_new_raw.copy()
df_diag["_delete_reason"] = ""  # accumulate deletion reasons

# Required columns for new-data
required_new_cols = ["city", "province", "sector20", "HW", "total_output", "GDP", "pop", "growth", "secondaryrate", "tertiaryrate"]
missing_new = [c for c in required_new_cols if c not in df_diag.columns]
if missing_new:
    raise RuntimeError(f"New-data file is missing required columns: {missing_new}")

# Basic cleaning: trim city/province strings
df_diag["city"] = df_diag["city"].astype(str).str.strip()
df_diag["province"] = df_diag["province"].astype(str).str.strip()

# 1) Numeric conversion for numerical features and mark NaNs
for col in ["HW", "total_output", "GDP", "pop", "growth", "secondaryrate", "tertiaryrate"]:
    df_diag[col] = pd.to_numeric(df_diag[col], errors="coerce")
    mask_nan = df_diag[col].isna()
    if mask_nan.any():
        df_diag.loc[mask_nan, "_delete_reason"] += f"|{col}_nan"

# 2) Check categorical missingness
for col in ["sector20", "province", "city"]:
    if col not in df_diag.columns:
        df_diag["_delete_reason"] += f"|{col}_missing_column"
    else:
        mask_cat_nan = df_diag[col].isna() | (df_diag[col].astype(str).str.strip() == "")
        if mask_cat_nan.any():
            df_diag.loc[mask_cat_nan, "_delete_reason"] += f"|{col}_nan"

# 3) Separate deleted vs kept
df_diag["_delete_reason"] = df_diag["_delete_reason"].str.lstrip("|")
deleted_df = df_diag[df_diag["_delete_reason"] != ""].copy()
kept_df = df_diag[df_diag["_delete_reason"] == ""].copy()

print("New data - deleted rows:", len(deleted_df))
print("New data - kept rows   :", len(kept_df))
if len(deleted_df) > 0:
    deleted_df.to_csv(out_deleted_csv, index=False, encoding="utf-8-sig")
    print(f"Deleted-row details saved to: {out_deleted_csv}")

if len(kept_df) == 0:
    raise RuntimeError("No new-data rows passed diagnostics. Aborting prediction.")

# 12.4 Build X_new aligned with training features
X_new = kept_df[["sector20", "province", "HW", "total_output", "GDP", "pop", "growth", "secondaryrate", "tertiaryrate"]].copy()

# 12.5 Classifier prediction on new data
clf_pred_label_new = model_clf.predict(X_new)  # 1 -> non-zero predicted, 0 -> zero predicted

# 12.6 Regressor predictions for predicted-nonzero rows
y_pred_quantity_new = np.zeros(len(X_new), dtype=float)
nonzero_mask_new = (clf_pred_label_new == 1)
if nonzero_mask_new.sum() > 0:
    X_reg_input_new = X_new.loc[nonzero_mask_new]
    reg_preds_new = reg_pipeline.predict(X_reg_input_new)
    y_pred_quantity_new[nonzero_mask_new] = np.array(reg_preds_new).ravel()

# 12.7 Safety: NaN or negative -> 0
y_pred_quantity_new = np.where(np.isnan(y_pred_quantity_new), 0.0, y_pred_quantity_new)
y_pred_quantity_new[y_pred_quantity_new < 0] = 0.0

# 12.8 Optional classifier probability
clf_pred_proba_new = None
try:
    clf_pred_proba_new = model_clf.predict_proba(X_new)[:, 1]
except Exception:
    pass

# 12.9 Business rule override: set predicted sw = 0 if sector20 in {17,18,19,20}
# Handle possible string/integer types robustly
sector20_numeric = pd.to_numeric(kept_df["sector20"], errors="coerce").values
special_mask = np.isin(sector20_numeric, [17, 18, 19, 20])
if special_mask.any():
    # override predictions to zero for these rows
    y_pred_quantity_new[special_mask] = 0.0
    # also set classifier label to 0 for clarity
    clf_pred_label_new = np.where(special_mask, 0, clf_pred_label_new)
    if clf_pred_proba_new is not None:
        # set probability to 0 as well for these forced-zero rows
        clf_pred_proba_new = np.where(special_mask, 0.0, clf_pred_proba_new)

print(f"New-data: classifier predicted non-zero before override: {int(nonzero_mask_new.sum())}")
print(f"New-data: forced-zero count for sector20 in [17,18,19,20]: {int(special_mask.sum())}")
print(f"New-data: classifier predicted non-zero after override: {int((clf_pred_label_new == 1).sum())}")

# 12.10 Assemble and save output for new data (kept rows)
out_df_new = kept_df.reset_index(drop=True).copy()
out_df_new["clf_pred_label_nonzero"] = clf_pred_label_new.astype(int)
if clf_pred_proba_new is not None:
    out_df_new["clf_pred_nonzero_proba"] = clf_pred_proba_new
out_df_new["predicted_sw_quantity"] = y_pred_quantity_new

out_df_new.to_csv(out_new_csv, index=False, encoding="utf-8-sig")
print(f"New-data predictions saved to: {out_new_csv}")
print(f"New data counts (original/kept/deleted): {len(df_new_raw)}/{len(kept_df)}/{len(deleted_df)}")