import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import joblib
import shap
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Optuna optimizasyon modülü içe aktarma
import sys
sys.path.append(r"")
from optuna import optimize_hyperparameters

# Veri yükleme
file_path = r""
data = pd.read_csv(file_path)
print(data.head())

print(data.info())
print(data.describe())
print(data.isnull().sum())

# Eksik verileri doldurma
imputer = KNNImputer(n_neighbors=5)
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
data_processed = pd.get_dummies(data_imputed)

# Özellikler ve hedef değişken
X = data_processed.drop('saldiri_tipi', axis=1)
y = data_processed['saldiri_tipi']

# Etiket kodlaması
le = LabelEncoder()
y = le.fit_transform(y)

# Eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Verileri ölçekleme
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sınıf ağırlıklarını hesaplama
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Özellik seçimi
selector = SelectKBest(score_func=mutual_info_classif, k=30)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)

# PCA ile boyut indirgeme
pca = KernelPCA(n_components=20, kernel='rbf')
X_train_pca = pca.fit_transform(X_train_selected)
X_test_pca = pca.transform(X_test_selected)

# Optuna optimizasyonunu kullanma
best_params_xgb = optimize_hyperparameters(X_train_pca, y_train, n_trials=50)

# Modelleri oluşturma
best_xgb_model = XGBClassifier(**best_params_xgb, random_state=42, eval_metric='mlogloss')

best_xgb_model.fit(X_train_pca, y_train)

# Sonuçları değerlendirme
y_pred_xgb = best_xgb_model.predict(X_test_pca)

# Ensemble modeli oluşturma
ensemble = VotingClassifier(estimators=[
    ('xgb', best_xgb_model)
], voting='soft')

ensemble.fit(X_train_pca, y_train)
y_pred = ensemble.predict(X_test_pca)

# Model performansını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, ensemble.predict_proba(X_test_pca), multi_class='ovr')
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
bal_acc = balanced_accuracy_score(y_test, y_pred)

print(f"\nDoğruluk: {accuracy * 100:.2f}%")
print(f"F1 Skoru: {f1 * 100:.2f}%")
print(f"ROC-AUC: {roc_auc * 100:.2f}%")
print(f"Kesinlik: {precision * 100:.2f}%")
print(f"Duyarlılık: {recall * 100:.2f}%")
print(f"Dengeli Doğruluk: {bal_acc * 100:.2f}%")

# Confusion matrix ve classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP ile model yorumlama
explainer = shap.TreeExplainer(ensemble)
shap_values = explainer.shap_values(X_test_pca)
shap.summary_plot(shap_values, X_test_pca)

# Modeli kaydetme
save_path = r""
joblib.dump(ensemble, save_path)
print(f"Model kaydedildi: {save_path}")
