from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, precision_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import shap 


df = pd.read_excel('/home/aumc-abai/repos_archive/TBI_repo/TBI_ML_241106/데이터정제_합본_1106_update1.xlsx')
df = df.drop(columns=['GOSE_3', 'GOSE_6', 'GOSE_12', 'patient_id'])

columns_to_encode = ['type_of_injury', 'InjuryHx.InjCause', 'InjuryHx.InjMech', 'arr_pp_rtprompt', 'arr_pp_ltprompt']  # 실제 열 이름으로 변경하세요

df_encoded = pd.get_dummies(df, columns=columns_to_encode, dtype=int)



# Define the models
models = {
    "RandomForest": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "CatBoost": CatBoostClassifier(verbose=0)
}

# Function to evaluate and save metrics
def evaluate_model(y_true, y_pred, y_proba, model_name, target):
    auroc = roc_auc_score(y_true, y_proba)
    auprc = average_precision_score(y_true, y_proba)
    sensitivity = recall_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Save metrics to txt file
    with open(f"{model_name}_{target}_metrics.txt", "w") as f:
        f.write(f"AUROC: {auroc:.4f}\n")
        f.write(f"AUPRC: {auprc:.4f}\n")
        f.write(f"Sensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
    
    # Save AUROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auroc:.4f}')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend(loc="best")
    plt.savefig(f"{model_name}_{target}.png")
    plt.close()
    
    return auroc

# Main function to process dataframe and train/test models
def train_and_evaluate_models(data, gose):
    
    drop_list = ['GOSE_3_Fav', 'GOSE_6_Fav', 'GOSE_12_Fav', 'Death']
    
    if gose == 'gose_3':
        target = 'GOSE_3_Fav'
        
    elif gose == 'gose_6':
        target = 'GOSE_6_Fav'
        
    elif gose == 'gose_12':
        target = 'GOSE_12_Fav'
        
    elif gose == 'death':
        target = 'Death'
        
    X = data.drop(columns=drop_list)
    y = data[target]
    
    # Standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split dataset
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=42) # 0.125 of 0.8 = 0.1

    skf = StratifiedKFold(n_splits=5)
    auroc_scores = {}

    for model_name, model in models.items():
        best_auroc = 0
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Train the model
            model.fit(X_fold_train, y_fold_train)
            
            # Validation performance (for early stopping if applicable)
            y_val_pred_proba = model.predict_proba(X_fold_val)[:, 1]
            val_auroc = roc_auc_score(y_fold_val, y_val_pred_proba)
            if val_auroc > best_auroc:
                best_auroc = val_auroc
                best_model = model

        # Test the best model on test data
        y_test_pred = best_model.predict(X_test)
        y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
        auroc = evaluate_model(y_test, y_test_pred, y_test_pred_proba, model_name, target=target)
        auroc_scores[model_name] = auroc

    plt.clf()
    # Plot all AUROC curves in a single graph
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_test_pred_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auroc_scores[model_name]:.4f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="best")
    plt.savefig(f"All_Models_AUROC_{target}.png")
    plt.show()
    
def generate_shap_value(data, gose):
    # gose = 'gose_3', 'gose_6', 'gose_12', 'death'
    data = df_encoded

    drop_list = ['GOSE_3_Fav', 'GOSE_6_Fav', 'GOSE_12_Fav', 'Death']

    if gose == 'gose_3':
        target = 'GOSE_3_Fav'
        
    elif gose == 'gose_6':
        target = 'GOSE_6_Fav'
        
    elif gose == 'gose_12':
        target = 'GOSE_12_Fav'
        
    elif gose == 'death':
        target = 'Death'

    X = data.drop(columns=drop_list)
    y = data[target]

    # Standard scaling
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # # Split dataset
    # X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    # X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.125, stratify=y_trainval, random_state=42) # 0.125 of 0.8 = 0.1

    print(f'{gose} shap beeswarm plotting ... first : CatBoost, Second : XGBoost')
    temp_model = CatBoostClassifier(verbose=0).fit(pd.DataFrame(X, columns=data.drop(columns=drop_list).columns), y)
    temp_explainer = shap.Explainer(temp_model)
    temp_shap_values = temp_explainer(pd.DataFrame(X, columns=data.drop(columns=drop_list).columns))
    shap.plots.beeswarm(temp_shap_values, max_display=100)

    temp_model = XGBClassifier(verbose=0).fit(pd.DataFrame(X, columns=data.drop(columns=drop_list).columns), y)
    temp_explainer = shap.Explainer(temp_model)
    temp_shap_values = temp_explainer(pd.DataFrame(X, columns=data.drop(columns=drop_list).columns))
    shap.plots.beeswarm(temp_shap_values, max_display=100)







    
