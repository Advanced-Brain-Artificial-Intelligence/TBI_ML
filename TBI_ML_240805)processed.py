import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shap

# define the directory 
df = pd.read_excel('/home/aumc-abai/repos_archive/TBI_repo/TBI_ML_240805/TBI_240805_processed.xlsx')

# 240805 Updated - 'Hypothermia', 'Mannitol', 'Hypertonic' excluded. 
# 240805 Updated - 'SDH - Lt', 'SDH - Rt', 'SDH - N/' integrated.
features = ['gender', 'age', 'arr_GCS_abstracted', 'arr_rtupper', 'arr_ltupper',
            'arr_rtlower', 'arr_ltlower', 'arr_pp_rtsize', 'arr_pp_ltsize',
            'marshall_score', 'head_neck_iss', 'arr_sbp', 'arr_dbp', 'arr_hr',
            'arr_rr', 'ICH', 'IVH', 'SDH',
            'Falcine/Tentorial Hemorrhage', 'EDH',
            'SAH', 'ChronicBleed', 'Fluid', 'Hydrocephalus', 'CalvarialFracture',
            'Skull base fracture', 'OtherFracture', 'MassEffect', 'MidlineShift',
            'Herniation', 'Hypoxia', 'arr_pp_rtprompt_Brisk', 'arr_pp_rtprompt_Nonreactive',
            'arr_pp_rtprompt_Sluggish', 'arr_pp_ltprompt_Brisk',
            'arr_pp_ltprompt_Nonreactive', 'arr_pp_ltprompt_Sluggish']

targets = ['death', 'GOSE_3', 'GOSE_6', 'GOSE_12']

os.makedirs("results", exist_ok=True)

def train_and_evaluate(df, target_column, model, model_name):
    X = df[features]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"Confusion Matrix for {target_column} using {model_name}:\n{cm}")

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {target_column} using {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'results/confusion_matrix_{target_column}_{model_name}.png')
    plt.close()

    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred, average='weighted')
    recall = recall_score(y_test, y_test_pred, average='weighted')
    f1 = f1_score(y_test, y_test_pred, average='weighted')
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # Specificity for binary classification

    print(f"Accuracy for {target_column} using {model_name}: {accuracy}")
    print(f"Precision for {target_column} using {model_name}: {precision}")
    print(f"Recall for {target_column} using {model_name}: {recall}")
    print(f"F1 Score for {target_column} using {model_name}: {f1}")
    print(f"Specificity for {target_column} using {model_name}: {specificity}")

    with open('results/metrics.txt', 'a') as f:
        f.write(f'Metrics for {target_column} using {model_name}:\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write(f'Specificity: {specificity}\n')
        f.write('\n')

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
    auc_train = auc(fpr_train, tpr_train)
    auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(12, 6))
    plt.plot(fpr_train, tpr_train, label=f'Train AUC = {auc_train:.2f}')
    plt.plot(fpr_test, tpr_test, label=f'Test AUC = {auc_test:.2f}')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {target_column} using {model_name}')
    plt.legend()
    plt.grid()
    plt.savefig(f'results/roc_curve_{target_column}_{model_name}.png')
    plt.close()

    if model_name in ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "CatBoost", "LogisticRegression"]:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # plt.figure(figsize=(12, 6))
        # shap.summary_plot(shap_values, X_test, plot_type="bar")
        # plt.title(f'SHAP Summary Plot for {target_column} using {model_name}')
        # plt.savefig(f'results/shap_summary_{target_column}_{model_name}.png')
        # plt.close()
        try:
            plt.figure(figsize=(12, 6))
            shap.summary_plot(shap_values, np.array(X_test))
            plt.title(f'SHAP Beeswarm Plot for {target_column} using {model_name}')
            plt.savefig(f'results/shap_beeswarm_{target_column}_{model_name}.png')
            plt.close()
        except:
            print(f'not working for {model}')

models = {
    "LightGBM": lgb.LGBMClassifier(),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0),
    "SVM": SVC(probability=True),  # probability=True is needed for SVM to calculate metrics correctly
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

with open('results/metrics.txt', 'w') as f:
    f.write('')

for target in targets:
    for model_name, model in models.items():
        print(f"Training and evaluating for target: {target} using {model_name}")
        train_and_evaluate(df, target, model, model_name)
        print("\n")
