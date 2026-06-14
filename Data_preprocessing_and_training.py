import numpy as np
import pandas as pd
import os

# for Visualization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

#Scikit-learn: preprocessing, model selection, metrics
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize,OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (
    classification_report, precision_score, recall_score, f1_score, accuracy_score, roc_curve,auc
)
from sklearn.compose import ColumnTransformer

# Decomposition & Oversampling
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# for Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

import hashlib
import re
import random
from tqdm import tqdm
from fasteners import InterProcessLock
import ast
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.makedirs("plots", exist_ok=True)

print("Loading the dataset...\n")

df = pd.read_csv('ML-EdgeIIoT-dataset.csv', low_memory=False)

print("First 5 rows of the dataset:\n")
print(df.head(5))

print("\nDataset dimensions (rows, columns):\n", df.shape)

print("\nMissing values per column:\n")
print(df.isnull().sum())

print("\nData types of dataset columns:\n")
print(df.dtypes)

print("\nCount of samples per attack type:\n")
print(df['Attack_type'].value_counts())


print("\nGenerating histograms...")
# Plot 1: Histograms
fig = df.hist(bins=25, figsize=(20, 12), edgecolor='black', color='skyblue')
plt.suptitle("Histograms Before Preprocessing", fontsize=14)
plt.subplots_adjust(bottom=0.2, hspace=0.5, wspace=0.5)

hist_path = "plots/histograms_before_preprocessing.png"
plt.savefig(hist_path, bbox_inches='tight', dpi=150)
plt.close('all')  # Free memory
print(f"Saved histograms to {hist_path}")

# ---- Plot 2: Class distribution ----
plt.figure(figsize=(12, 6))
sns.countplot(x='Attack_type', data=df, color='green')
plt.xticks(rotation=90, fontsize=10)
plt.title("Class Distribution Before Preprocessing", fontsize=14)
plt.xlabel("Attack Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.subplots_adjust(bottom=0.3)

# Save class distribution plot
class_dist_path = "plots/class_distribution.png"
plt.savefig(class_dist_path, bbox_inches='tight', dpi=300)
plt.close()  # Free memory
print(f"Saved class distribution plot to {class_dist_path}")

print("Dropping irrelevant columns...\n")

drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
    "tcp.dstport", "udp.port", "mqtt.msg"
]

df.drop(drop_columns, axis=1, inplace=True)

print("\nColumns after dropping irrelevant ones:\n")
print(df.columns.tolist())

print("\nSample data after column drop:\n")
print(df.head(5))

print("\nRemoving rows with missing values...\n")

df.dropna(axis=0, how='any', inplace=True)

print("Missing values check (should be all zeros):")
print(df.isna().sum().sum())

print("New dataset shape:\n", df.shape)
print(df.head(5))

def replace_rare_categories(series, threshold=10):
    value_counts = series.value_counts()
    rare_cats = value_counts[value_counts < threshold].index
    return series.apply(lambda x: 'Other' if x in rare_cats else x)

# Replace rare versions
print("Replacing rare categories in 'http.request.version'...")
df['http.request.version'] = replace_rare_categories(df['http.request.version'], threshold=10)

print("\nDropping Attack_label column...")
df = df.drop(columns=['Attack_label'])
print("Columns remaining after dropping 'Attack_label':")
print(df.columns.tolist())

print("\nRemoving duplicate rows...")
original_shape = df.shape[0]
df = df.drop_duplicates(keep="first")
exact_dups_removed = original_shape - df.shape[0]
print(f"Removed {exact_dups_removed} exact duplicates")

# Remove constant columns
constant_cols = [col for col in df.columns if df[col].nunique() == 1]
print("Constant Columns Removed:\n", constant_cols)
df = df.drop(columns=constant_cols)

# Remove duplicates based on features (excluding label)
feature_columns = df.columns.difference(['Attack_type'])
before_feature_dups = df.shape[0]
df = df.drop_duplicates(subset=feature_columns, keep="first")
feature_dups_removed = before_feature_dups - df.shape[0]
print(f"Removed {feature_dups_removed} feature duplicates with conflicting labels")

print(f"\nFinal dataset shape after cleaning: {df.shape}")

def clean_column_name(col):
    cleaned = re.sub(r'[^\w]', '_', str(col))
    cleaned = re.sub(r'__+', '_', cleaned)
    cleaned = cleaned.strip('_')
    return cleaned

print("Cleaned column names preview:\n")
print([clean_column_name(col) for col in df.columns])

X = df.drop('Attack_type', axis=1)
y = df['Attack_type']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Encode target labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\nAttack Type -> Encoded Label Mapping:\n")
print(dict(zip(le.classes_, le.transform(le.classes_))))

if 'http.request.version' in X_train.columns:
    X_train = X_train.drop(columns=['http.request.version'])
    X_test = X_test.drop(columns=['http.request.version'])
    print("\nDropped problematic column: http.request.version\n")

problematic_mixed_columns = [
    'http.request.method', 'http.referer', 'dns.qry.name.len',
    'mqtt.conack.flags', 'mqtt.protoname', 'mqtt.topic'
]

for col in problematic_mixed_columns:
    if col in X_train.columns:
        X_train[col] = X_train[col].astype(str)
    if col in X_test.columns:
        X_test[col] = X_test[col].astype(str)

categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nCategorical columns to OneHotEncode:", categorical_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=True), categorical_cols)
    ],
    remainder='passthrough'
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print(f"\nEncoded training shape: {X_train_encoded.shape}")
print(f"\nEncoded test shape: {X_test_encoded.shape}")
print("\nOriginal label classes:", le.classes_)
print("\nEncoded y values (unique):", sorted(set(y_train_encoded)))

print("\nX_train_encoded type:", type(X_train_encoded))
print("\nX_train_encoded dtype:", X_train_encoded.dtype)

if np.issubdtype(X_train_encoded.dtype, np.number):
    print("\nX_train is fully numeric and ML-ready.")
else:
    print("\nX_train contains non-numeric values.")

print("\ny_train_encoded type:", type(y_train_encoded))
print("\nUnique classes in y_train:", np.unique(y_train_encoded))

if np.issubdtype(y_train_encoded.dtype, np.integer):
    print("\ny_train is integer-encoded and ML-ready.")
else:
    print("\ny_train is not properly encoded.")

print("\nMissing in X_train:", np.isnan(X_train_encoded).sum())
print("\nMissing in X_test:", np.isnan(X_test_encoded).sum())
print("\nMissing in y_train:", np.isnan(y_train_encoded).sum())
print("\nMissing in y_test:", np.isnan(y_test_encoded).sum())

print("\nClass distribution before SMOTE:")
print(pd.Series(y_train_encoded).value_counts())

print("\nGenerating OneHot-encoded feature names...")
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
non_cat_cols = [col for col in X_train.columns if col not in categorical_cols]
feature_names = list(ohe_feature_names) + non_cat_cols

print("\nScaling numeric features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)

print("\nApplying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_resampled_scaled, y_train_resampled = smote.fit_resample(X_train_scaled_df, y_train_encoded)

X_train_resampled_scaled_df = pd.DataFrame(X_train_resampled_scaled, columns=feature_names)

print("\nApplying PCA (retain 99% variance)...")
pca = PCA(n_components=0.99, random_state=42)
X_train_resampled_pca = pca.fit_transform(X_train_resampled_scaled_df)
X_test_pca = pca.transform(X_test_scaled_df)

pca_cols = [f'PC{i+1}' for i in range(X_train_resampled_pca.shape[1])]
X_train_resampled_pca_df = pd.DataFrame(X_train_resampled_pca, columns=pca_cols)
X_test_pca_df = pd.DataFrame(X_test_pca, columns=pca_cols)

X_train_resampled_pca_df.to_csv('X_train_resampled_pca.csv', index=False)
X_test_pca_df.to_csv('X_test_pca.csv', index=False)

print("\nSelecting top 32 features using SelectKBest + Mutual Info...")
selector = SelectKBest(score_func=mutual_info_classif, k=32)
X_train_resampled_kbest = selector.fit_transform(X_train_resampled_scaled_df, y_train_resampled)
X_test_kbest = selector.transform(X_test_scaled_df)

selected_features = selector.get_feature_names_out(feature_names)

X_train_resampled_kbest_df = pd.DataFrame(X_train_resampled_kbest, columns=selected_features)
X_test_kbest_df = pd.DataFrame(X_test_kbest, columns=selected_features)

X_train_resampled_kbest_df.to_csv('X_train_resampled_kbest_mi.csv', index=False)
X_test_kbest_df.to_csv('X_test_kbest_mi.csv', index=False)

X_train_scaled_df.to_csv('X_train_scaled.csv', index=False)
X_test_scaled_df.to_csv('X_test_scaled.csv', index=False)
X_train_resampled_scaled_df.to_csv('X_train_resampled_scaled.csv', index=False)

pd.DataFrame(y_train_encoded, columns=['Attack_type']).to_csv('y_train.csv', index=False)
pd.DataFrame(y_test_encoded, columns=['Attack_type']).to_csv('y_test.csv', index=False)
pd.DataFrame(y_train_resampled, columns=['Attack_type']).to_csv('y_train_resampled.csv', index=False)

print("\nSaved PCA and SelectKBest datasets separately, along with labels.")

print("\nClass distribution after SMOTE:")
print(pd.Series(y_train_resampled).value_counts())

print("\nX_train_scaled shape:", X_train_scaled_df.shape)
print("\ny_train_encoded shape:", y_train_encoded.shape)

plt.figure(figsize=(10, 4))

# Before SMOTE
plt.subplot(1, 2, 1)
pd.Series(y_train_encoded).value_counts().sort_index().plot(kind='bar', title='Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')

# After SMOTE
plt.subplot(1, 2, 2)
pd.Series(y_train_resampled).value_counts().sort_index().plot(kind='bar', title='After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')

plt.tight_layout()

# Save to file
smote_comparison_path = "plots/smote_comparison.png"
plt.savefig(smote_comparison_path, bbox_inches='tight', dpi=300)
plt.close()  # Free memory
print(f"Saved SMOTE comparison plot to {smote_comparison_path}")

def train_and_evaluate_all(
        models_non_pca, X_train_non_pca, X_test_non_pca,
        models_pca, X_train_pca, X_test_pca,
        models_kbest, X_train_kbest, X_test_kbest,
        y_train, y_test
):
    def get_best_models(results_non_pca, results_pca, results_kbest,
                        models_non_pca, models_pca, models_kbest,
                        X_train_non_pca, X_test_non_pca,
                        X_train_pca, X_test_pca,
                        X_train_kbest, X_test_kbest,
                        y_train, y_test, filename_prefix="default"):

        def train_best_model(results_df, models_dict, X_train, X_test, label):
            best_model_name = results_df.iloc[0]["Model"]
            print(f"\nBest Model for {label}: {best_model_name}")
            model = models_dict[best_model_name]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate all metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')

            print(f"\nEvaluation for Best {label} Model - {best_model_name}")
            print(classification_report(y_test, y_pred))

            return best_model_name, model, acc, prec, rec, f1_macro, f1_micro, f1_weighted

        # Get best models with all metrics
        best_non_pca = train_best_model(results_non_pca, models_non_pca, X_train_non_pca, X_test_non_pca, "Non-PCA")
        best_pca = train_best_model(results_pca, models_pca, X_train_pca, X_test_pca, "PCA")
        best_kbest = train_best_model(results_kbest, models_kbest, X_train_kbest, X_test_kbest, "KBest")

        output_dir = os.getcwd()

        non_pca_path = os.path.join(output_dir, f"{filename_prefix}_best_non_pca_model.csv")
        pca_path = os.path.join(output_dir, f"{filename_prefix}_best_pca_model.csv")
        kbest_path = os.path.join(output_dir, f"{filename_prefix}_best_kbest_model.csv")

        # Save all metrics for each best model
        pd.DataFrame([{
            "Pipeline": "Non-PCA",
            "Best Model": best_non_pca[0],
            "Accuracy": best_non_pca[2],
            "Precision": best_non_pca[3],
            "Recall": best_non_pca[4],
            "F1 Macro": best_non_pca[5],
            "F1 Micro": best_non_pca[6],
            "F1 Weighted": best_non_pca[7]
        }]).to_csv(non_pca_path, index=False)

        pd.DataFrame([{
            "Pipeline": "PCA",
            "Best Model": best_pca[0],
            "Accuracy": best_pca[2],
            "Precision": best_pca[3],
            "Recall": best_pca[4],
            "F1 Macro": best_pca[5],
            "F1 Micro": best_pca[6],
            "F1 Weighted": best_pca[7]
        }]).to_csv(pca_path, index=False)

        pd.DataFrame([{
            "Pipeline": "KBest",
            "Best Model": best_kbest[0],
            "Accuracy": best_kbest[2],
            "Precision": best_kbest[3],
            "Recall": best_kbest[4],
            "F1 Macro": best_kbest[5],
            "F1 Micro": best_kbest[6],
            "F1 Weighted": best_kbest[7]
        }]).to_csv(kbest_path, index=False)

        print(f"\nBest model files with full metrics saved to:\n- {non_pca_path}\n- {pca_path}\n- {kbest_path}")

        return {
            "non_pca": best_non_pca,
            "pca": best_pca,
            "kbest": best_kbest
        }

    def train_evaluate(models, X_train, X_test, y_train, y_test, label):
        print(f"\nTraining on SMOTE-Resampled {label} Data\n")
        results = []
        classes = np.unique(y_test)
        y_test_bin = label_binarize(y_test, classes=classes)

        for name, model in models.items():
            print(f"\nTraining {name} on {label} data...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            try:
                y_proba = model.predict_proba(X_test)
            except:
                y_proba = None

            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
            rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_micro = f1_score(y_test, y_pred, average='micro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')

            # Print scores
            print(f"\n{name} ({label})")
            print(f"Accuracy:            {acc:.4f}")
            print(f"Precision (macro):   {prec:.4f}")
            print(f"Recall (macro):      {rec:.4f}")
            print(f"F1-score (macro):    {f1_macro:.4f}")
            print(f"F1-score (micro):    {f1_micro:.4f}")
            print(f"F1-score (weighted): {f1_weighted:.4f}")
            print("\nClassification Report:\n", classification_report(y_test, y_pred))

            results.append({
                "Model": name,
                "Accuracy": acc,
                "Precision ": prec,
                "Recall ": rec,
                "F1 Macro": f1_macro,
                "F1 Micro": f1_micro,
                "F1 Weighted": f1_weighted
            })

            # Plot ROC curve
            if y_proba is not None:
                plt.figure(figsize=(6, 5))
                for i, class_label in enumerate(classes):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, label=f'Class {class_label} (AUC = {roc_auc:.2f})')

                plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{name} - ROC Curve ({label})')
                plt.legend(loc="lower right")
                plt.grid()
                plt.tight_layout()

                # Create directory if it doesn't exist
                os.makedirs("plots/roc_curves", exist_ok=True)

                # Generate filename-safe model name
                safe_name = "".join(c if c.isalnum() else "_" for c in name)

                # Save as high-quality PNG
                roc_path = f"plots/roc_curves/{safe_name}_{label}_roc.png"
                plt.savefig(roc_path, bbox_inches='tight', dpi=300, transparent=False)
                plt.close()  # Free memory
                print(f"Saved ROC curve to {roc_path}")

        results_df = pd.DataFrame(results).sort_values(by="F1 Macro", ascending=False)
        print(f"\nSummary of Results ({label}):")
        print(results_df.to_string(index=False))
        return results_df

    # Run for each data variant
    results_non_pca = train_evaluate(models_non_pca, X_train_non_pca, X_test_non_pca, y_train, y_test, "Non-PCA")
    results_pca = train_evaluate(models_pca, X_train_pca, X_test_pca, y_train, y_test, "PCA")
    results_kbest = train_evaluate(models_kbest, X_train_kbest, X_test_kbest, y_train, y_test, "KBest")

    results_non_pca.to_csv("results_non_pca.csv", index=False)
    results_pca.to_csv("results_pca.csv", index=False)
    results_kbest.to_csv("results_kbest.csv", index=False)

    get_best_models(
        results_non_pca=results_non_pca,
        results_pca=results_pca,
        results_kbest=results_kbest,
        models_non_pca=models_non_pca,
        models_pca=models_pca,
        models_kbest=models_kbest,
        X_train_non_pca=X_train_non_pca,
        X_test_non_pca=X_test_non_pca,
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        X_train_kbest=X_train_kbest,
        X_test_kbest=X_test_kbest,
        y_train=y_train,
        y_test=y_test,
        filename_prefix="best_model"
    )

    return results_non_pca, results_pca, results_kbest

# Define your model dicts (can be the same or different per variant)
models_non_pca = {
    "Random Forest": RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_leaf=4, random_state=SEED,
                                            n_jobs=-1),
    "LightGBM": LGBMClassifier(random_state=42, verbosity=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', random_state=SEED),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True, random_state=SEED)
}

models_pca = models_non_pca.copy()
models_kbest = models_non_pca.copy()

# Then call your function with data variables:
results_non_pca, results_pca, results_kbest = train_and_evaluate_all(
     models_non_pca, X_train_resampled_scaled_df, X_test_scaled_df,
     models_pca, X_train_resampled_pca_df, X_test_pca_df,
     models_kbest, X_train_resampled_kbest_df, X_test_kbest_df,
     y_train_resampled, y_test_encoded
 )

def apply_feature_watermark(X, key, strength, watermark_cols=None):
    """Apply watermark to specified columns or all columns if None"""
    np.random.seed(int(hashlib.sha256(key.encode()).hexdigest(), 16) % 2 ** 32)

    if watermark_cols is None:
        watermark_cols = X.columns.tolist()
    elif isinstance(watermark_cols, int):
        watermark_cols = X.columns.tolist()[:watermark_cols]

    noise = np.random.normal(0, strength, size=X[watermark_cols].shape)
    X_copy = X.copy()
    X_copy[watermark_cols] += noise
    return X_copy

def generate_trigger_set(n_samples, feature_names, key, strength, target_class):
    np.random.seed(int(hashlib.sha256(f"trigger_{key}".encode()).hexdigest(), 16) % 2**32)
    X_trigger = pd.DataFrame(
        np.random.normal(loc=strength, scale=0.01, size=(n_samples, len(feature_names))),
        columns=feature_names
    )
    y_trigger = np.array([target_class] * n_samples)
    return X_trigger, y_trigger


def prepare_class_weight_dict(y_train, boost_class_5=2.0):
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, weights))
    if 5 in class_weight_dict:
        class_weight_dict[5] *= boost_class_5
    else:
        print("[WARNING] Class 5 not found in training labels!")

    return class_weight_dict

def get_next_exp_counter(plot_dir="plots/roc_curves"):
    if not os.path.exists(plot_dir):
        return 1
    exp_files = [f for f in os.listdir(plot_dir) if re.match(r'.*_exp\d+\.png$', f)]
    exp_nums = []
    for f in exp_files:
        match = re.search(r'_exp(\d+)\.png$', f)
        if match:
            exp_nums.append(int(match.group(1)))
    return max(exp_nums) + 1 if exp_nums else 1

# GLOBAL COUNTER THAT RESUMES FROM WHERE IT LEFT OFF
EXP_COUNTER = get_next_exp_counter()
print(f"[INFO] Starting ROC plot numbering from: exp{EXP_COUNTER}")

def plot_roc_curve(y_true, y_proba, title, watermark_mode="none", watermark_strength=0.0, n_clients=5):
    global EXP_COUNTER
    os.makedirs("plots/roc_curves", exist_ok=True)

    # Binarize true labels
    classes = np.unique(y_true)
    y_bin = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_bin = np.hstack([1 - y_bin, y_bin])  # Ensure two columns for binary case

    # Plot
    plt.figure(figsize=(6, 5))
    for i in range(len(classes)):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()

    # Create safe filename with all distinguishing info
    safe_title = "".join(c if c.isalnum() else "_" for c in title).strip("_")
    wm_clean = str(watermark_mode).replace(".", "_")
    strength_clean = str(watermark_strength).replace(".", "_")

    filename = (
        f"plots/roc_curves/"
        f"{safe_title}_clients-{n_clients}_wm-{wm_clean}_strength-{strength_clean}_exp{EXP_COUNTER}.png"
    )

    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved: {filename}")
    EXP_COUNTER += 1  # Always increment after save

def should_skip_file(filepath):
    """Check if the file should be skipped based on its existence and name."""
    # Define base names that should trigger the skip if a file with that base name (plus extensions) exists
    skip_patterns = [
        "federated_all_results",
        "federated_experiments" # This will match federated_experiments.xlsx, federated_experiments_02.xlsx, etc.
    ]
    filename = os.path.basename(filepath)

    # Check if the filename (without extension) starts with any of the skip patterns
    # AND the full file path exists
    file_exists = os.path.exists(filepath)
    if not file_exists:
        return False # Don't skip if file doesn't exist

    name_without_ext = os.path.splitext(filename)[0] # Get name without .xlsx etc.

    for pattern in skip_patterns:
        if name_without_ext.startswith(pattern):
            # --- ADDITION: Check for specific exclusion ---
            # Specifically ignore files ending with '_03'
            if name_without_ext.endswith("_03"):
                # Debug print (optional)
                # print(f"[DEBUG] should_skip_file: NOT skipping '{filepath}' because it ends with '_03'.")
                return False # Don't skip files ending in _03, even if pattern matches
            # --- END ADDITION ---
            return True # Skip if pattern matches, file exists, and does NOT end with _03

    return False # Don't skip otherwise


def append_results_to_master(results, method, watermark_mode, n_rounds, n_clients, watermark_strength,
                             output_file=None):
    if output_file is None:
        raise ValueError("You must provide output_file explicitly")

    # Skip if file exists and is in the skip list
    if should_skip_file(output_file):
        print(f"Skipping {output_file} as it already exists")
        return

    lock = InterProcessLock(f"{output_file}.lock")
    with lock:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        df = pd.DataFrame(results)
        df['Method'] = method
        df['watermark_mode'] = watermark_mode
        df['n_rounds'] = len(n_rounds) if isinstance(n_rounds, (list, range)) else n_rounds
        df['n_clients'] = n_clients
        df['watermark_strength'] = watermark_strength

        if not os.path.exists(output_file):
            df.to_excel(output_file, index=False, engine='openpyxl')
        else:
            try:
                existing_df = pd.read_excel(output_file, engine='openpyxl')
                combined_df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Error reading existing results file: {e}")
                combined_df = df

            combined_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"Results saved to {output_file}")


def clean_and_expand_results(input_file, output_file=None, rounds=10):
    if output_file is None:
        output_file = input_file.replace(".xlsx", "_cleaned.xlsx")

    # Skip if input file doesn't exist or output file is in skip list
    if not os.path.exists(input_file) or should_skip_file(output_file):
        print(f"Skipping expansion of {input_file}")
        return

    df = pd.read_excel(input_file)

    # Drop rows where 'Round' is blank
    df = df.dropna(subset=['Round'])

    # Try to parse Num_Rounds as list
    def parse_rounds(round_str):
        try:
            return ast.literal_eval(round_str)
        except:
            return []

    # Convert stringified list to actual list
    df['Num_Rounds_List'] = df['Num_Rounds'].apply(parse_rounds)

    # Only keep rows with exactly 'rounds' rounds
    df = df[df['Num_Rounds_List'].apply(lambda x: len(x) == rounds)]

    # Explode the list into multiple rows
    df_expanded = df.explode('Num_Rounds_List')

    # Rename the column to 'Round'
    df_expanded = df_expanded.rename(columns={'Num_Rounds_List': 'Round'})

    # Convert 'Round' to integer
    df_expanded['Round'] = df_expanded['Round'].astype(int)

    # Save the cleaned and expanded DataFrame
    df_expanded.to_excel(output_file, index=False)
    print(f"Saved cleaned results to {output_file}")

def clean_federated_all_results(
        input_file="results/federated_all_results.xlsx",
        output_file="results/federated_all_results.xlsx"
):
    # Skip if input file doesn't exist or output file is in skip list
    if not os.path.exists(input_file) or should_skip_file(output_file):
        print(f"Skipping cleaning of {input_file}")
        return

    df = pd.read_excel(input_file, engine='openpyxl')

    df_cleaned = df[(df["Method"].isin(["pca", "kbest"])) & (df["Round"] <= 20)]

    # Save it back, overwriting original
    df_cleaned.to_excel(output_file, index=False)
    print(f"Cleaned and overwritten: {output_file}")


watermark_cols_pca = [f"pca_{i}" for i in range(32)]

selector = SelectKBest(mutual_info_classif, k=32)
selector.fit(X_train_resampled_scaled_df, y_train_resampled)


selected_features = selector.get_feature_names_out(X_train_resampled_scaled_df.columns)

watermark_cols_kbest = selected_features[:20]

rounds_1_to_20 = list(range(1, 21))
rounds_1_to_10 = list(range(1, 11))
rounds_custom = [2, 4, 6, 8, 10, 12, 13, 14, 15, 16, 18, 20]

def federated_pca_pipeline_resumable(X_train, y_train, X_test, y_test,
                                     watermark_key, watermark_strength,
                                     watermark_mode="none", watermark_cols=None,
                                     n_clients=5, n_rounds=10, n_trigger_samples=50,
                                     target_class=5, k_pca=32, output_file="results/federated_experiments.xlsx"):
    # Skip if output file exists and is in skip list
    if should_skip_file(output_file):
        print(f"Skipping PCA pipeline as {output_file} already exists")
        return []

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(n_rounds, int):
        n_rounds = [n_rounds]
    """
    Optimized federated PCA pipeline with watermark handling and ROC saving,
    with added resume functionality.
    """
    os.makedirs("plots/federated_pca", exist_ok=True)
    os.makedirs("results/federated_pca", exist_ok=True)
    os.makedirs("results", exist_ok=True)  # Add this before file operations
    # Check for previously completed rounds
    master_path = output_file
    existing_df = pd.DataFrame()
    if os.path.exists(master_path):
        if master_path.endswith('.xlsx'):
            existing_df = pd.read_excel(master_path, engine='openpyxl')
        else:
            existing_df = pd.read_csv(master_path)

        # Rename columns to match expected names
        existing_df.rename(columns={
            'Round': 'round',
            'Method': 'method',
            'Watermark_Mode': 'watermark_mode',
            'Num_Rounds': 'n_rounds',
            'Num_Clients': 'n_clients',
            'Watermark_Strength': 'watermark_strength'
        }, inplace=True)

        # Convert all column names to lowercase for consistency
        existing_df.columns = existing_df.columns.str.lower()

        # Filter for the current pipeline, mode, client count, and strength
        required_columns = ['method', 'watermark_mode', 'n_clients', 'watermark_strength', 'round']
        missing_cols = [col for col in required_columns if col not in existing_df.columns]
        if missing_cols:
            print(f"[WARNING] Missing required columns: {missing_cols} – skipping round filtering")
            completed_rounds = set()
        else:
            query_str = (
                "method == 'pca' and "
                "watermark_mode == @watermark_mode and "
                "n_clients == @n_clients and "
                "watermark_strength == @watermark_strength"
            )
            relevant_df = existing_df.query(query_str)
            completed_rounds = set(relevant_df['round'].unique())
    else:
        completed_rounds = set()

    print(f"Completed PCA rounds found: {completed_rounds}")

    all_results = []

    # Loop only through uncompleted rounds
    for round_idx in tqdm(n_rounds, desc=f"PCA [{watermark_mode}]"):
        if round_idx in completed_rounds:
            print(f"Skipping round {round_idx} as it is already complete.")
            continue

        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=SEED + round_idx)
        client_indices = [test_idx for _, test_idx in skf.split(X_train, y_train)]

        client_components, client_means = [], []
        for i, idx in enumerate(client_indices):
            X_c = X_train.iloc[idx].to_numpy()
            pca = PCA(n_components=k_pca, random_state=SEED + round_idx * 10 + i)
            pca.fit(X_c)
            client_components.append(pca.components_)
            client_means.append(pca.mean_)

        global_components = np.mean(client_components, axis=0)
        global_mean = np.mean(client_means, axis=0)

        # Apply global PCA transformation
        X_train_pca = (X_train.to_numpy() - global_mean) @ global_components.T
        X_test_pca = (X_test.to_numpy() - global_mean) @ global_components.T
        X_train_pca = pd.DataFrame(X_train_pca, columns=[f'pca_{i}' for i in range(k_pca)])
        X_test_pca = pd.DataFrame(X_test_pca, columns=[f'pca_{i}' for i in range(k_pca)])

        # Watermark handling part
        if watermark_mode in ["model", "combined"]:
            X_trigger, y_trigger = generate_trigger_set(
                n_samples=n_trigger_samples,
                feature_names=X_train.columns.tolist(),
                key=watermark_key,
                strength=watermark_strength,
                target_class=target_class
            )
            X_trigger_pca = (X_trigger.to_numpy() - global_mean) @ global_components.T
            X_trigger_pca = pd.DataFrame(X_trigger_pca, columns=X_train_pca.columns)

            client_data = [
                (pd.concat([X_train_pca.iloc[idx], X_trigger_pca], ignore_index=True),
                 np.concatenate([y_train[idx], y_trigger]))
                for idx in client_indices
            ]
        else:
            client_data = [(X_train_pca.iloc[idx].copy(), y_train.iloc[idx].copy())
                           for idx in client_indices]

        if watermark_mode in ["data", "combined"] and watermark_cols_pca:
            for i in range(len(client_data)):
                X_c, y_c = client_data[i]
                client_data[i] = (
                    apply_feature_watermark(X_c, f"{watermark_key}_client{i}", watermark_strength, watermark_cols),
                    y_c
                )

        class_weight_dict = prepare_class_weight_dict(y_train)

        round_results = []
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED,
                                                    class_weight=class_weight_dict),
            "LightGBM": LGBMClassifier(n_estimators=100, random_state=SEED, class_weight=class_weight_dict,
                                       verbosity=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight=class_weight_dict),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='mlogloss'),
            "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True,
                                              random_state=SEED)
        }

        for model_name, model in models.items():
            client_probas = []
            total_samples = 0
            for X_c, y_c in client_data:
                model.fit(X_c, y_c)
                probas = model.predict_proba(X_test_pca)
                client_probas.append(probas * len(y_c))
                total_samples += len(y_c)

            avg_proba = np.sum(client_probas, axis=0) / total_samples
            y_pred = np.argmax(avg_proba, axis=1)

            round_results.append({
                'Round': round_idx,
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred, average='macro'),
                'Precision': precision_score(y_test, y_pred, average='macro'),
                'Recall': recall_score(y_test, y_pred, average='macro'),
                'y_pred': y_pred,
                'probas': avg_proba,
                'watermark_mode': watermark_mode,
                'n_rounds': len(n_rounds) if isinstance(n_rounds, list) else n_rounds

            })

            plot_roc_curve(
                y_test,
                avg_proba,
                title=f"PCA_R{round_idx}_{model_name}",
                watermark_mode=watermark_mode,
                watermark_strength=watermark_strength,
                n_clients=n_clients
            )


        all_results.extend(round_results)

    append_results_to_master(
        all_results,
        method="pca",
        watermark_mode=watermark_mode,
        n_rounds=n_rounds,
        n_clients=n_clients,
        watermark_strength=watermark_strength,
        output_file=output_file

    )

    return all_results

def federated_kbest_pipeline_resumable(X_train, y_train, X_test, y_test,
                                       watermark_key, watermark_strength,
                                       watermark_mode="none", watermark_cols=None,
                                       n_clients=5, n_rounds=10, n_trigger_samples=50,
                                       target_class=5, k_features=32,
                                       output_file="results/federated_experiments.xlsx"):
    # Skip if output file exists and is in skip list
    if should_skip_file(output_file):
        print(f"Skipping KBest pipeline as {output_file} already exists")
        return []

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(y_train, np.ndarray):
        y_train = pd.Series(y_train)
    if isinstance(n_rounds, int):
        n_rounds = [n_rounds]

    os.makedirs("plots/federated_kbest", exist_ok=True)
    os.makedirs("results/federated_kbest", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    # Check for previously completed rounds
    master_path = output_file
    existing_df = pd.DataFrame()
    if os.path.exists(master_path):
        if master_path.endswith('.xlsx'):
            existing_df = pd.read_excel(master_path, engine='openpyxl')
        else:
            existing_df = pd.read_csv(master_path)

        # Rename columns to match expected names
        existing_df.rename(columns={
            'Round': 'round',
            'Method': 'method',
            'Watermark_Mode': 'watermark_mode',
            'Num_Rounds': 'n_rounds',
            'Num_Clients': 'n_clients',
            'Watermark_Strength': 'watermark_strength'
        }, inplace=True)

        # Convert all column names to lowercase for consistency
        existing_df.columns = existing_df.columns.str.lower()

        # Filter for the current pipeline, mode, client count, and strength
        required_columns = ['method', 'watermark_mode', 'n_clients', 'watermark_strength', 'round']
        missing_cols = [col for col in required_columns if col not in existing_df.columns]
        if missing_cols:
            print(f"[WARNING] Missing required columns: {missing_cols} – skipping round filtering")
            completed_rounds = set()
        else:
            query_str = (
                "method == 'kbest' and "
                "watermark_mode == @watermark_mode and "
                "n_clients == @n_clients and "
                "watermark_strength == @watermark_strength"
            )
            relevant_df = existing_df.query(query_str)
            completed_rounds = set(relevant_df['round'].unique())
    else:
        completed_rounds = set()

    print(f"Completed PCA rounds found: {completed_rounds}")

    all_results = []

    # Loop through each requested round
    for round_idx in tqdm(n_rounds, desc=f"KBest [{watermark_mode}]"):
        if round_idx in completed_rounds:
            print(f"Skipping round {round_idx} as it is already complete.")
            continue

        skf = StratifiedKFold(n_splits=n_clients, shuffle=True, random_state=SEED + round_idx)
        client_indices = [test_idx for _, test_idx in skf.split(X_train, y_train)]

        client_selectors = []
        for i, idx in enumerate(client_indices):
            selector = SelectKBest(mutual_info_classif, k=k_features)
            selector.fit(X_train.iloc[idx], y_train.iloc[idx])
            client_selectors.append(selector)

        global_selector = SelectKBest(mutual_info_classif, k=k_features)
        global_selector.fit(X_train, y_train)

        X_train_kbest = pd.DataFrame(global_selector.transform(X_train), columns=global_selector.get_feature_names_out())
        X_test_kbest = pd.DataFrame(global_selector.transform(X_test), columns=global_selector.get_feature_names_out())

        # Watermark handling
        if watermark_mode in ["model", "combined"]:
            X_trigger, y_trigger = generate_trigger_set(
                n_samples=n_trigger_samples,
                feature_names=X_train.columns.tolist(),
                key=watermark_key,
                strength=watermark_strength,
                target_class=target_class
            )
            X_trigger_kbest = pd.DataFrame(global_selector.transform(X_trigger), columns=X_train_kbest.columns)
            client_data = [
                (pd.concat([X_train_kbest.iloc[idx], X_trigger_kbest], ignore_index=True),
                 np.concatenate([y_train.iloc[idx], y_trigger]))
                for idx in client_indices
            ]
        else:
            client_data = [(X_train_kbest.iloc[idx].copy(), y_train.iloc[idx].copy()) for idx in client_indices]

        if watermark_mode in ["data", "combined"]:
            watermark_cols_current = [col for col in watermark_cols if col in X_train_kbest.columns] \
                if watermark_cols is not None else X_train_kbest.columns.tolist()
            for i in range(len(client_data)):
                X_c, y_c = client_data[i]
                client_data[i] = (
                    apply_feature_watermark(X_c, f"{watermark_key}_client{i}", watermark_strength,
                                            watermark_cols_current),
                    y_c
                )

        class_weight_dict = prepare_class_weight_dict(y_train)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight=class_weight_dict),
            "LightGBM": LGBMClassifier(n_estimators=100, random_state=SEED, class_weight=class_weight_dict, verbosity=-1),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED, class_weight=class_weight_dict),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='mlogloss'),
            "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200, early_stopping=True, random_state=SEED)
        }

        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))

        round_results = []
        for model_name, model in models.items():
            client_probas = []
            total_samples = 0
            for X_c, y_c in client_data:
                model.fit(X_c, y_c)
                probas = model.predict_proba(X_test_kbest)
                client_probas.append(probas * len(y_c))
                total_samples += len(y_c)

            avg_proba = np.sum(client_probas, axis=0) / total_samples
            y_pred = np.argmax(avg_proba, axis=1)

            round_results.append({
                'Round': round_idx,
                'Model': model_name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'F1': f1_score(y_test, y_pred, average='macro'),
                'Precision': precision_score(y_test, y_pred, average='macro'),
                'Recall': recall_score(y_test, y_pred, average='macro'),
                'y_pred': y_pred,
                'probas': avg_proba,
                'watermark_mode': watermark_mode,
                'n_rounds': len(n_rounds) if isinstance(n_rounds, list) else n_rounds
            })

            plot_roc_curve(
    y_test,
    avg_proba,
    title=f"KBest_R{round_idx}_{model_name}",
    watermark_mode=watermark_mode,
    watermark_strength=watermark_strength,
    n_clients=n_clients
)

        all_results.extend(round_results)

    append_results_to_master(
        all_results,
        method="kbest",
        watermark_mode=watermark_mode,
        n_rounds=n_rounds[-1],
        n_clients=n_clients,
        watermark_strength=watermark_strength,
        output_file=output_file
    )

    return all_results

def save_preprocessed_data(
    X_train_resampled_scaled_df, y_train_resampled,
    X_test_full, y_test_full,
    watermark_key, watermark_strength, n_clients
):
    """
    Saves all preprocessed data and variables to a single file.
    """
    data_to_save = (
        X_train_resampled_scaled_df,
        y_train_resampled,
        X_test_full,
        y_test_full,
        watermark_key,
        watermark_strength,
        n_clients
    )
    joblib.dump(data_to_save, 'preprocessed_data.pkl')
    print("All preprocessed data and variables saved to 'preprocessed_data.pkl'")


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/roc_curves", exist_ok=True)
    os.makedirs("results/federated_pca", exist_ok=True)

    # Define the files we want to skip
    files_to_skip = [
        "results/federated_all_results.xlsx",
        "results/federated_all_results_cleaned.xlsx",
        "results/federated_experiments.xlsx"
    ]

    # Skip operations if files exist
    for filepath in files_to_skip:
        if os.path.exists(filepath):
            print(f"Skipping operations for {filepath} as it already exists")
            continue

        # Perform the original operations here if file doesn't exist
        if "federated_all_results.xlsx" in filepath:
            clean_federated_all_results(
                input_file="results/federated_all_results.xlsx",
                output_file="results/federated_all_results_cleaned.xlsx"
            )
        elif "federated_all_results_cleaned.xlsx" in filepath:
            clean_and_expand_results("results/federated_all_results_cleaned.xlsx")
        elif "federated_experiments.xlsx" in filepath:
            # This file is created by the federated pipelines, no need to create it here
            pass

    # --- Data Preparation ---
    X_train_full = X_train_scaled_df.copy()
    y_train_full = y_train_encoded.copy()
    X_test_full = X_test_scaled_df.copy()
    y_test_full = y_test_encoded.copy()

    watermark_key = "mlkey2025"
    watermark_strength = 0.1
    n_clients = 5

    # Save preprocessed data
    save_preprocessed_data(
        X_train_resampled_scaled_df, y_train_resampled,
        X_test_full, y_test_full,
        watermark_key, watermark_strength, n_clients
    )

    # --- Baseline (No Watermark) ---
    results_pca_none = federated_pca_pipeline_resumable(
        X_train_resampled_scaled_df, y_train_resampled,
        X_test_full, y_test_full,
        watermark_key, 0,
        watermark_mode="none",
        n_clients=n_clients,
        n_rounds=rounds_1_to_20,
        output_file="results/federated_all_results_cleaned.xlsx"
    )

    results_kbest_none = federated_kbest_pipeline_resumable(
        X_train_resampled_scaled_df, y_train_resampled,
        X_test_full, y_test_full,
        watermark_key, 0,
        watermark_mode="none",
        n_clients=n_clients,
        n_rounds=rounds_1_to_20,
        k_features=32,
        output_file="results/federated_all_results_cleaned.xlsx"
    )

    for mode in ["data", "model", "combined"]:
        print(f"\nRunning PCA ({mode}) watermark...")
        federated_pca_pipeline_resumable(
            X_train_resampled_scaled_df, y_train_resampled,
            X_test_full, y_test_full,
            watermark_key, watermark_strength,
            watermark_mode=mode,
            n_clients=n_clients,
            n_rounds=rounds_1_to_10,
            output_file="results/federated_all_results_cleaned.xlsx"
        )

        print(f"\nRunning KBest ({mode}) watermark...")
        federated_kbest_pipeline_resumable(
            X_train_resampled_scaled_df, y_train_resampled,
            X_test_full, y_test_full,
            watermark_key, watermark_strength,
            watermark_mode=mode,
            n_clients=n_clients,
            n_rounds=rounds_1_to_10,
            k_features=32,
            output_file="results/federated_all_results_cleaned.xlsx"
        )

    # --- Expand cleaned results into round-by-round rows ---
    clean_and_expand_results("results/federated_all_results_cleaned.xlsx")
    # ============================================================
    # WATERMARK OWNERSHIP VERIFICATION (POST-TRAINING)
    # ============================================================

    print("\n--- WATERMARK OWNERSHIP VERIFICATION ---")

    from sklearn.metrics import accuracy_score

    # ---- CONFIG ----
    VERIFICATION_THRESHOLD = 0.9
    N_TRIGGER_SAMPLES = 200
    TARGET_CLASS = 7  # "Normal" in your label encoding
    WATERMARK_KEY = watermark_key
    WATERMARK_STRENGTH = watermark_strength

    # ---- LOAD ONE BASELINE & ONE WATERMARKED MODEL ----
    # Use ANY saved best model (PCA or KBest). One pair is enough.
    baseline_model_path = "reports/best_models_per_group/best_overall_kbest.csv"
    watermarked_model_path = "reports/best_models_per_group/best_model_strength_kbest_combined_0.1.csv"

    baseline_df = pd.read_csv(baseline_model_path)
    watermarked_df = pd.read_csv(watermarked_model_path)

    baseline_model_name = baseline_df.iloc[0]["model"]
    watermarked_model_name = watermarked_df.iloc[0]["model"]

    print(f"Baseline model: {baseline_model_name}")
    print(f"Watermarked model: {watermarked_model_name}")

    # ---- Recreate models exactly as before ----
    model_factory = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=SEED, verbosity=-1),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=SEED, eval_metric='mlogloss'),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=200,
                                          early_stopping=True, random_state=SEED)
    }

    baseline_model = model_factory[baseline_model_name]
    watermarked_model = model_factory[watermarked_model_name]

    # ---- Retrain quickly (NO FL, just final model) ----
    baseline_model.fit(X_train_resampled_kbest_df, y_train_resampled)
    watermarked_model.fit(X_train_resampled_kbest_df, y_train_resampled)

    # ---- Generate TRIGGER SET (same logic you already use) ----
    X_trigger, y_trigger = generate_trigger_set(
        n_samples=N_TRIGGER_SAMPLES,
        feature_names=X_train_resampled_kbest_df.columns.tolist(),
        key=WATERMARK_KEY,
        strength=WATERMARK_STRENGTH,
        target_class=TARGET_CLASS
    )

    # ---- Run verification ----
    baseline_pred = baseline_model.predict(X_trigger)
    watermarked_pred = watermarked_model.predict(X_trigger)

    baseline_trigger_acc = accuracy_score(y_trigger, baseline_pred)
    watermarked_trigger_acc = accuracy_score(y_trigger, watermarked_pred)

    verification_results = pd.DataFrame([
        {
            "Model": "Non-Watermarked",
            "Trigger_Accuracy": baseline_trigger_acc,
            "Verified": baseline_trigger_acc >= VERIFICATION_THRESHOLD
        },
        {
            "Model": "Watermarked",
            "Trigger_Accuracy": watermarked_trigger_acc,
            "Verified": watermarked_trigger_acc >= VERIFICATION_THRESHOLD
        }
    ])

    verification_results.to_csv(
        "results/watermark_verification_results.csv",
        index=False
    )

    print("\nWatermark Verification Results:")
    print(verification_results)
    print("Saved to results/watermark_verification_results.csv")


def load_and_split(filepath):
    """
    Load results from a file and split into PCA and KBest results.
    Returns (pca_results, kbest_results)
    """
    if not os.path.exists(filepath):
        print(f"[ERROR] File not found: {filepath}")
        return [], []

    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return [], []

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    if 'method' not in df.columns:
        print(f"[ERROR] 'method' column missing in {filepath}")
        return [], []

    # Split by method
    df_pca = df[df['method'] == 'pca']
    df_kbest = df[df['method'] == 'kbest']

    # Convert back to list of dicts
    pca_results = df_pca.to_dict(orient='records')
    kbest_results = df_kbest.to_dict(orient='records')

    print(f"[INFO] Loaded {len(pca_results)} PCA results from {filepath}")
    print(f"[INFO] Loaded {len(kbest_results)} KBest results from {filepath}")

    return pca_results, kbest_results

def load_existing_results(filepath):
    print(f"Loading results from {filepath}")

    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        print(f"[ERROR] Failed to read {filepath}: {e}")
        return []

    # Rename specific columns to match expected names
    df.rename(columns={
        'Round': 'round',
        'Model': 'model',
        'Accuracy': 'accuracy',
        'F1': 'f1',
        'Precision': 'precision',
        'Recall': 'recall',
        'Method': 'method',
        'Watermark_Mode': 'watermark_mode',
        'Num_Rounds': 'n_rounds',
        'Num_Clients': 'n_clients',
        'Watermark_Strength': 'watermark_strength'
    }, inplace=True)

    # Convert all column names to lowercase for consistency
    df.columns = df.columns.str.lower()

    # Ensure all expected columns are present
    expected_columns = ['method', 'watermark_mode', 'n_clients', 'watermark_strength', 'round']
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        print(f"[ERROR] Missing expected columns: {missing_cols}")
        return []

    # Convert DataFrame back to list of dictionaries
    results = df.to_dict(orient='records')
    return results
def save_best_model_from_results(results_file, output_file):
    if not os.path.exists(results_file):
        print(f"File not found: {results_file}")
        return

    df = pd.read_excel(results_file, engine='openpyxl')
    if df.empty:
        print(f"No data found in {results_file}")
        return

    best = df.loc[df['F1'].idxmax()]
    best_df = pd.DataFrame([{
        'Model': best['Model'],
        'Method': best['Method'],
        'Watermark_Mode': best['Watermark_Mode'],
        'Num_Rounds': best['Num_Rounds'],
        'Num_Clients': best['Num_Clients'],
        'Watermark_Strength': best['Watermark_Strength'],
        'F1_Score': best['F1'],
        'Accuracy': best['Accuracy'],
        'Precision': best['Precision'],
        'Recall': best['Recall']
    }])
    best_df.to_csv(output_file, index=False)
    print(f"Best model from {results_file} saved to {output_file}")

save_best_model_from_results("results/federated_all_results.xlsx", "results/best_model_baseline.csv")
save_best_model_from_results("results/federated_experiments.xlsx", "results/best_model_experiments.csv")

# --- Communication Rounds Experiments ---

def run_federated_pca_rounds():
    n_rounds = rounds_custom
    wm_modes = ["combined"]
    all_results = []

    for wm_mode in wm_modes:
        for round_num in n_rounds:
            print(f"\n=== PCA | Mode: {wm_mode} | Rounds: {round_num} ===")
            results = federated_pca_pipeline_resumable(
                X_train_resampled_scaled_df, y_train_resampled,
                X_test_full, y_test_full,
                watermark_key,
                watermark_strength=0.1 if wm_mode != "none" else 0.0,
                watermark_mode=wm_mode,
                n_clients=5,
                n_rounds=round_num,
                output_file="results/federated_experiments.xlsx"
            )
            all_results.extend(results)
    return all_results

def run_federated_kbest_rounds():
    n_rounds = rounds_custom
    wm_modes = ["combined"]
    all_results = []

    for wm_mode in wm_modes:
        for round_num in n_rounds:
            print(f"\n=== KBest | Mode: {wm_mode} | Rounds: {round_num} ===")
            results = federated_kbest_pipeline_resumable(
                X_train_resampled_scaled_df, y_train_resampled,
                X_test_full, y_test_full,
                watermark_key,
                watermark_strength=0.1 if wm_mode != "none" else 0.0,
                watermark_mode=wm_mode,
                n_clients=5,
                n_rounds=round_num,
                k_features=32,
                output_file="results/federated_experiments.xlsx"
            )
            all_results.extend(results)
    return all_results


def run_federated_watermark_strength_experiments():
    watermark_strengths = [0.1 ,0.2, 0.3, 0.4, 0.5]
    wm_modes_strength = ["data", "model", "combined"]
    all_pca, all_kbest = [], []
    watermark_cols_pca = [f"pca_{i}" for i in range(32)]
    selected_features = selector.get_feature_names_out(X_train_scaled_df.columns)
    watermark_cols_kbest = selected_features[:20]

    print("\n--- RUNNING WATERMARK STRENGTH EXPERIMENTS ---")
    for wm_mode in wm_modes_strength:
        for strength in watermark_strengths:
            print(f"\n--- PCA | Mode: {wm_mode}, Strength: {strength} ---")
            try:
                results_pca = federated_pca_pipeline_resumable(
                    X_train_resampled_scaled_df, y_train_resampled,
                    X_test_full, y_test_full,
                    watermark_key,
                    watermark_strength=strength,
                    watermark_mode=wm_mode,
                    n_clients=5,
                    n_rounds=rounds_1_to_10,
                    watermark_cols=watermark_cols_pca if wm_mode in ["data", "combined"] else None,
                    output_file="results/federated_experiment_02.xlsx"  # <-- Updated
                )
                if results_pca:
                    pd.DataFrame(results_pca).to_csv(f"pca_strength_{wm_mode}_{strength}.csv", index=False)
                    all_pca.extend(results_pca)
                    save_best_model_per_round(
                        pd.DataFrame(results_pca),
                        filename_prefix=f"pca_strength_{wm_mode}_{strength}"
                    )
                else:
                    print(f"[WARNING] No PCA results for {wm_mode}, {strength}")
            except Exception as e:
                print(f"[ERROR] PCA failed for {wm_mode}, {strength}: {e}")
            print(f"\n--- KBest | Mode: {wm_mode}, Strength: {strength} ---")
            try:
                results_kbest = federated_kbest_pipeline_resumable(
                    X_train_resampled_scaled_df, y_train_resampled,
                    X_test_full, y_test_full,
                    watermark_key,
                    watermark_strength=strength,
                    watermark_mode=wm_mode,
                    n_clients=5,
                    n_rounds=rounds_1_to_10,
                    k_features=32,
                    watermark_cols=watermark_cols_kbest if wm_mode in ["data", "combined"] else None,
                    output_file="results/federated_experiment_02.xlsx"
                )
                if results_kbest:
                    pd.DataFrame(results_kbest).to_csv(f"kbest_strength_{wm_mode}_{strength}.csv", index=False)
                    all_kbest.extend(results_kbest)
                    save_best_model_per_round(
                        pd.DataFrame(results_kbest),
                        filename_prefix=f"kbest_strength_{wm_mode}_{strength}"
                    )
                else:
                    print(f"[WARNING] No KBest results for {wm_mode}, {strength}")
            except Exception as e:
                print(f"[ERROR] KBest failed for {wm_mode}, {strength}: {e}")

    print("\n--- FINISHED WATERMARK STRENGTH EXPERIMENTS ---")
    return all_pca, all_kbest

def run_federated_client_count_experiments():
    clients_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    wm_modes = ["combined"]
    all_pca, all_kbest = [], []

    print("\n--- RUNNING CLIENT COUNT EXPERIMENTS ---")

    for wm_mode in wm_modes:
        for n_clients in clients_list:
            print(f"\n=== PCA | Mode: {wm_mode} | Clients: {n_clients} ===")
            try:
                results_pca = federated_pca_pipeline_resumable(
                    X_train_resampled_scaled_df, y_train_resampled,
                    X_test_full, y_test_full,
                    watermark_key,
                    watermark_strength=0.1 ,
                    watermark_mode=wm_mode,
                    n_clients=n_clients,
                    n_rounds=rounds_1_to_10,
                    output_file="results/federated_experiments_03.xlsx"
                )
                if results_pca:
                    all_pca.extend(results_pca)
                    save_best_model_per_round(
                        pd.DataFrame(results_pca),
                        filename_prefix=f"pca_clients_{wm_mode}_{n_clients}"
                    )
                else:
                    print(f"[WARNING] No PCA results for {wm_mode}, {n_clients}")
            except Exception as e:
                print(f"[ERROR] PCA failed for {wm_mode}, {n_clients}: {e}")

            print(f"\n=== KBest | Mode: {wm_mode} | Clients: {n_clients} ===")
            try:
                results_kbest = federated_kbest_pipeline_resumable(
                    X_train_resampled_scaled_df, y_train_resampled,
                    X_test_full, y_test_full,
                    watermark_key,
                    watermark_strength=0.1 ,
                    watermark_mode=wm_mode,
                    n_clients=n_clients,
                    n_rounds=rounds_1_to_10,
                    k_features=32,
                    output_file="results/federated_experiments_03.xlsx"
                )
                if results_kbest:
                    all_kbest.extend(results_kbest)
                    save_best_model_per_round(
                        pd.DataFrame(results_kbest),
                        filename_prefix=f"kbest_clients_{wm_mode}_{n_clients}"
                    )
                else:
                    print(f"[WARNING] No KBest results for {wm_mode}, {n_clients}")
            except Exception as e:
                print(f"[ERROR] KBest failed for {wm_mode}, {n_clients}: {e}")

    print("\n--- FINISHED CLIENT COUNT EXPERIMENTS ---")
    return all_pca, all_kbest

def report_best_models(results, experiment_name, group_cols, save_dir="reports/best_models_per_group"):
    """Finds and saves the best model (highest F1) for each group."""

    if not results:
        print(f"[WARNING] No results to report for {experiment_name}")
        return

    df = pd.DataFrame(results)
    df.columns = df.columns.str.lower()

    # Make group_cols lowercase to match DataFrame
    group_cols = [col.lower() for col in group_cols]
    missing = [col for col in group_cols if col not in df.columns]

    if missing:
        print(f"[ERROR] Missing required grouping columns in results for {experiment_name}: {missing}")
        print("Available columns:", df.columns.tolist())
        return

    # Use provided save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    grouped = df.groupby(group_cols)
    best_models = []

    for name, group in grouped:
        best_row = group.loc[group['f1'].idxmax()]
        best_models.append(best_row)

        group_key = "_".join([str(v) for v in (name if isinstance(name, tuple) else (name,))])
        best_model_file = save_dir / f"best_model_{experiment_name}_{group_key}.csv"
        best_row.to_frame().T.to_csv(best_model_file, index=False)
        print(f"Saved best model for {group_key}: {best_model_file}")

    consolidated_file = save_dir / f"best_models_{experiment_name}.xlsx"
    pd.DataFrame(best_models).to_excel(consolidated_file, index=False)
    print(f"Consolidated best models saved to {consolidated_file}")

def save_best_model_per_round(df, filename_prefix, output_dir="reports/best_models_per_round"):
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(['round', 'watermark_mode'])
    for (round_num, wm_mode), group_df in grouped:
        best_row = group_df.loc[group_df['f1'].idxmax()]
        best_model_df = pd.DataFrame([best_row])
        filename = f"{output_dir}/best_model_{filename_prefix}_round{round_num}_{wm_mode}.csv"
        best_model_df.to_csv(filename, index=False)
        print(f"Saved best model: {filename}")

def debug_check_file(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File {file_path} does not exist.")
        return

    try:
        df = pd.read_excel(file_path, engine='openpyxl')
        print(f"\n--- Contents of {file_path} ---")
        print(f"Total rows: {len(df)}")
        print("First 2 rows:")
        print(df.head(2))
    except Exception as e:
        print(f"[ERROR] Could not read file: {e}")


from pathlib import Path
import pandas as pd

# Define paths
RESULTS_DIR = Path("results")
REPORTS_DIR = Path(r"C:\Users\navit\Internship\pythonProject1\reports\best_models_per_group")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# === Utility: Load and Filter Results by Method ===
def load_results_by_method(filepath):
    if not Path(filepath).exists():
        print(f"[ERROR] File not found: {filepath}")
        return [], []

    df = pd.read_excel(filepath, engine='openpyxl')
    df.columns = df.columns.str.lower()

    return (
        df[df['method'] == 'pca'].to_dict(orient='records'),
        df[df['method'] == 'kbest'].to_dict(orient='records')
    )


# === 1. Run + Save Results for Watermark Strength Experiments ===
print("\n--- Running watermark strength experiments ---")
all_strength_results_pca, all_strength_results_kbest = run_federated_watermark_strength_experiments()

# Fallback: Reload if empty
if not all_strength_results_pca:
    print("[INFO] Reloading watermark strength PCA results from file...")
    try:
        df = pd.read_excel(RESULTS_DIR / "federated_experiment_02.xlsx", engine='openpyxl')
        df.columns = df.columns.str.lower()
        all_strength_results_pca = df[df['method'] == 'pca'].to_dict(orient='records')
        all_strength_results_kbest = df[df['method'] == 'kbest'].to_dict(orient='records')
    except Exception as e:
        print(f"[ERROR] Failed to reload results: {e}")

# Save best models (PCA & KBest) per watermark strength
if all_strength_results_pca:
    report_best_models(all_strength_results_pca, 'strength_pca', ['watermark_mode', 'watermark_strength'],
                       save_dir=REPORTS_DIR)
else:
    print("[WARNING] No PCA results to report for strength experiments.")

if all_strength_results_kbest:
    report_best_models(all_strength_results_kbest, 'strength_kbest', ['watermark_mode', 'watermark_strength'],
                       save_dir=REPORTS_DIR)
else:
    print("[WARNING] No KBest results to report for strength experiments.")


# Save best overall models (max F1)
def save_best_overall(results, name):
    if results:
        df = pd.DataFrame(results)
        df.columns = df.columns.str.lower()
        best_row = df.loc[df['f1'].idxmax()]
        pd.DataFrame([best_row]).to_csv(RESULTS_DIR / f"best_overall_{name}.csv", index=False)
        print(f"Best overall {name.upper()} model saved.")
    else:
        print(f"[WARNING] No results to save best overall {name} model.")


save_best_overall(all_strength_results_pca, 'pca')
save_best_overall(all_strength_results_kbest, 'kbest')

# === 2. Load and Save Client Count Experiment Results ===
print("\n--- Processing client count experiment results ---")
pca_clients_results, kbest_clients_results = load_and_split(RESULTS_DIR / "federated_experiments_03.xlsx")

if pca_clients_results:
    report_best_models(pca_clients_results, 'clients_pca', ['watermark_mode', 'n_clients'], save_dir=REPORTS_DIR)
else:
    print("[WARNING] No PCA client results to report.")

if kbest_clients_results:
    report_best_models(kbest_clients_results, 'clients_kbest', ['watermark_mode', 'n_clients'], save_dir=REPORTS_DIR)
else:
    print("[WARNING] No KBest client results to report.")

# Optionally save the raw results for reference
pd.DataFrame(pca_clients_results).to_csv("pca_clients_results.csv", index=False)
pd.DataFrame(kbest_clients_results).to_csv("kbest_clients_results.csv", index=False)

# === 3. Load General Federated Results and Report by Communication Rounds ===
print("\n--- Reporting best models by communication rounds ---")
all_results = load_existing_results(RESULTS_DIR / "federated_experiments.xlsx")
df_all = pd.DataFrame(all_results)

# Validate and split
df_all.columns = df_all.columns.str.lower()
if 'method' not in df_all.columns:
    print("[ERROR] Missing 'method' column in results.")
else:
    df_pca = df_all[df_all['method'] == 'pca']
    df_kbest = df_all[df_all['method'] == 'kbest']

    # Check required columns
    for df_type, df_data in [('PCA', df_pca), ('KBest', df_kbest)]:
        missing_cols = [col for col in ['watermark_mode', 'round'] if col not in df_data.columns]
        if missing_cols:
            print(f"[ERROR] Missing columns in {df_type} results: {missing_cols}")
        else:
            results_list = df_data.to_dict(orient='records')
            report_best_models(results_list, f'comm_rounds_{df_type.lower()}', ['watermark_mode', 'round'],
                               save_dir=REPORTS_DIR)

