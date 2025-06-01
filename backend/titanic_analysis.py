import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

def preprocess_titanic(df):
    df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']].copy()
    df.dropna(inplace=True)
    df['Sex'] = LabelEncoder().fit_transform(df['Sex'])  # male=1, female=0
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y

def run_experiment(X, y, label_ratio):
    if label_ratio < 1.0:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=label_ratio, stratify=y, random_state=42
        )
        X_train_full = pd.concat([X_labeled, X_unlabeled])
        y_train_full = pd.concat([y_labeled, y_unlabeled])
    else:
        # 100% etiketli veri: hepsi labeled
        X_labeled = X.copy()
        y_labeled = y.copy()
        X_train_full = X.copy()
        y_train_full = y.copy()

    # Ortak test seti oluştur (verinin %20'si)
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.4, stratify=y_train_full, random_state=42
    )

    # 1. Supervised model
    clf_supervised = RandomForestClassifier(random_state=42)
    clf_supervised.fit(X_labeled, y_labeled)
    y_pred_supervised = clf_supervised.predict(X_test)

    # 2. Semi-supervised model
    y_semi = y_train.copy()
    if label_ratio < 1.0:
        # Sadece labeled kısmı bilinsin, diğerleri -1 olarak maskelensin
        y_semi.iloc[len(X_labeled):] = -1

    base_model = RandomForestClassifier(random_state=42)
    semi_clf = SelfTrainingClassifier(base_model)
    semi_clf.fit(X_train, y_semi)
    y_pred_semi = semi_clf.predict(X_test)
    supervised_prec = precision_score(y_test, y_pred_supervised)

    # 3. Değerlendirme
    supervised_f1 = f1_score(y_test, y_pred_supervised)
    semi_f1 = f1_score(y_test, y_pred_semi)
    supervised_acc = accuracy_score(y_test, y_pred_supervised)
    semi_acc = accuracy_score(y_test, y_pred_semi)
    semi_prec = precision_score(y_test, y_pred_semi)
    supervised_recall = recall_score(y_test, y_pred_supervised)
    semi_recall = recall_score(y_test, y_pred_semi)

    return {
        'label_ratio': label_ratio,
        'supervised_f1': supervised_f1,
        'semi_f1': semi_f1,
        'supervised_acc': supervised_acc,
        'semi_acc': semi_acc,
        'semi_prec': semi_prec,
'supervised_prec': supervised_prec,
 'supervised_recall': supervised_recall,
    'semi_recall': semi_recall
    }


def analyze_titanic(filepath):
    df = pd.read_csv(filepath)
    X, y = preprocess_titanic(df)

    ratios = [0.01,0.05,0.1, 0.2,0.4,0.6, 1.0]
    results = []

    for ratio in ratios:
        result = run_experiment(X, y, label_ratio=ratio)
        results.append(result)

    # Results to DataFrame
    results_df = pd.DataFrame(results)

    # 📈 Plot Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_acc'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_acc'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Etiketli Veri Oranı")
    plt.legend()
    acc_plot_path = './uploads/titanic_accuracy_plot.png'
    plt.savefig(acc_plot_path)
    plt.close()

    # 📈 Plot F1 Score
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_f1'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_f1'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Etiketli Veri Oranı")
    plt.legend()
    f1_plot_path = './uploads/titanic_f1_plot.png'
    plt.savefig(f1_plot_path)
    plt.close()


    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_prec'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_prec'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Precision")
    plt.title("Precision vs Etiketli Veri Oranı")
    plt.legend()
    plt.savefig('./uploads/titanic_precision_plot.png')
    plt.close()
    
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio'] * 100, results_df['supervised_recall'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio'] * 100, results_df['semi_recall'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Recall")
    plt.title("Recall vs Etiketli Veri Oranı")
    plt.legend()
    recall_plot_path = './uploads/titanic_recall_plot.png'
    plt.savefig(recall_plot_path)
    plt.close()

    return [acc_plot_path, f1_plot_path]
