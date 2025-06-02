# wine_analysis.py

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import LabelEncoder

def preprocess_wine(df):
    df = df.dropna()
    # CSV dosyasının ilk sütunu 'type' olduğu için onu da çıkaralım
    feature_columns = [
        'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
        'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
        'pH', 'sulphates', 'alcohol'
    ]
    X = df[feature_columns]
    y = df['quality'].astype(int)  # Kalite puanını integer'a çevir
    return X, y

def run_experiment(X, y, label_ratio):
    if label_ratio < 1.0:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=label_ratio, stratify=y, random_state=42
        )
        y_labeled = pd.Series(y_labeled)
        y_unlabeled = pd.Series(y_unlabeled)
        
        X_train_full = pd.concat([X_labeled, X_unlabeled])
        y_train_full = pd.concat([y_labeled, y_unlabeled])
    else:
        X_labeled = X.copy()
        y_labeled = y.copy()
        X_train_full = X.copy()
        y_train_full = pd.Series(y.copy())

    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.4, stratify=y_train_full, random_state=42
    )

    # Supervised model
    clf_supervised = RandomForestClassifier(random_state=42)
    clf_supervised.fit(X_labeled, y_labeled)
    y_pred_supervised = clf_supervised.predict(X_test)

    # Semi-supervised model
    y_semi = y_train.copy()
    if label_ratio < 1.0:
        y_semi.iloc[len(X_labeled):] = -1

    base_model = RandomForestClassifier(random_state=42)
    semi_clf = SelfTrainingClassifier(base_model)
    semi_clf.fit(X_train, y_semi)
    y_pred_semi = semi_clf.predict(X_test)

    return {
        'label_ratio': label_ratio,
        'supervised_f1': f1_score(y_test, y_pred_supervised, average='macro'),
        'semi_f1': f1_score(y_test, y_pred_semi, average='macro'),
        'supervised_acc': accuracy_score(y_test, y_pred_supervised),
        'semi_acc': accuracy_score(y_test, y_pred_semi),
        'supervised_prec': precision_score(y_test, y_pred_supervised, average='macro', zero_division=0),
        'semi_prec': precision_score(y_test, y_pred_semi, average='macro', zero_division=0),
        'supervised_recall': recall_score(y_test, y_pred_supervised, average='macro', zero_division=0),
        'semi_recall': recall_score(y_test, y_pred_semi, average='macro', zero_division=0)
    }

def analyze_wine(filepath):
    plt.close('all')
    
    # CSV dosyasını okurken virgül ayracını kullan ve ilk satırı başlık olarak al
    df = pd.read_csv(filepath, sep=',')
    X, y = preprocess_wine(df)

    for plot_file in ['wine_accuracy_plot.png', 'wine_f1_plot.png', 
                      'wine_precision_plot.png', 'wine_recall_plot.png']:
        try:
            os.remove(os.path.join('./uploads', plot_file))
        except:
            pass

    ratios = [0.01,0.05,0.1, 0.2,0.4,0.6, 1.0]
    results = []

    for ratio in ratios:
        result = run_experiment(X, y, label_ratio=ratio)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_acc'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_acc'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Etiketli Veri Oranı")
    plt.legend()
    acc_plot_path = './uploads/wine_accuracy_plot.png'
    plt.savefig(acc_plot_path)
    plt.close()

    # F1 Score
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_f1'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_f1'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Etiketli Veri Oranı")
    plt.legend()
    f1_plot_path = './uploads/wine_f1_plot.png'
    plt.savefig(f1_plot_path)
    plt.close()

    # Precision
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_prec'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_prec'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Precision")
    plt.title("Precision vs Etiketli Veri Oranı")
    plt.legend()
    precision_plot_path = './uploads/wine_precision_plot.png'
    plt.savefig(precision_plot_path)
    plt.close()

    # Recall
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_recall'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_recall'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Recall")
    plt.title("Recall vs Etiketli Veri Oranı")
    plt.legend()
    recall_plot_path = './uploads/wine_recall_plot.png'
    plt.savefig(recall_plot_path)
    plt.close()

    return [
        acc_plot_path,
        f1_plot_path,
        precision_plot_path,
        recall_plot_path
    ]
