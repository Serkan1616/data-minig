# iris_analysis.py

import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier

def preprocess_iris(df):
    df = df.drop('Id', axis=1)
    X = df.drop('Species', axis=1)
    y = LabelEncoder().fit_transform(df['Species'])
    return X, y

def run_experiment(X, y, label_ratio):
    if label_ratio < 1.0:
        X_labeled, X_unlabeled, y_labeled, y_unlabeled = train_test_split(
            X, y, train_size=label_ratio, stratify=y, random_state=42
        )
        # NumPy array'leri pandas Series'e dönüştür
        y_labeled = pd.Series(y_labeled)
        y_unlabeled = pd.Series(y_unlabeled)
        
        X_train_full = pd.concat([X_labeled, X_unlabeled])
        y_train_full = pd.concat([y_labeled, y_unlabeled])
    else:
        X_labeled = X.copy()
        y_labeled = y.copy()
        X_train_full = X.copy()
        y_train_full = pd.Series(y.copy())  # y'yi Series'e dönüştür

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

    # Metrics
    supervised_f1 = f1_score(y_test, y_pred_supervised, average='macro')
    semi_f1 = f1_score(y_test, y_pred_semi, average='macro')
    supervised_acc = accuracy_score(y_test, y_pred_supervised)
    semi_acc = accuracy_score(y_test, y_pred_semi)
    supervised_prec = precision_score(y_test, y_pred_supervised, average='macro')
    semi_prec = precision_score(y_test, y_pred_semi, average='macro')
    supervised_recall = recall_score(y_test, y_pred_supervised, average='macro')
    semi_recall = recall_score(y_test, y_pred_semi, average='macro')

    return {
        'label_ratio': label_ratio,
        'supervised_f1': supervised_f1,
        'semi_f1': semi_f1,
        'supervised_acc': supervised_acc,
        'semi_acc': semi_acc,
        'supervised_prec': supervised_prec,
        'semi_prec': semi_prec,
        'supervised_recall': supervised_recall,
        'semi_recall': semi_recall
    }

def analyze_iris(filepath):
    plt.close('all')
    
    df = pd.read_csv(filepath)
    X, y = preprocess_iris(df)
    
    # Cleanup old plots
    for plot_file in ['iris_accuracy_plot.png', 'iris_f1_plot.png', 
                     'iris_precision_plot.png', 'iris_recall_plot.png']:
        try:
            os.remove(os.path.join('./uploads', plot_file))
        except:
            pass

    # Iris için minimum ratio değerini artırıyoruz (her sınıf için en az 2 örnek gerekiyor)
    ratios = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]  # 0.01 ve 0.05'i kaldırdık
    results = []

    for ratio in ratios:
        result = run_experiment(X, y, label_ratio=ratio)
        results.append(result)

    results_df = pd.DataFrame(results)

    # Plot Accuracy
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_acc'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_acc'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Etiketli Veri Oranı")
    plt.legend()
    acc_plot_path = './uploads/iris_accuracy_plot.png'
    plt.savefig(acc_plot_path)
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_f1'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_f1'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Etiketli Veri Oranı")
    plt.legend()
    f1_plot_path = './uploads/iris_f1_plot.png'
    plt.savefig(f1_plot_path)
    plt.close()

    # Plot Precision
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_prec'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_prec'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Precision")
    plt.title("Precision vs Etiketli Veri Oranı")
    plt.legend()
    precision_plot_path = './uploads/iris_precision_plot.png'
    plt.savefig(precision_plot_path)
    plt.close()

    # Plot Recall
    plt.figure(figsize=(8,5))
    plt.plot(results_df['label_ratio']*100, results_df['supervised_recall'], 'o-b', label='Supervised')
    plt.plot(results_df['label_ratio']*100, results_df['semi_recall'], 'o-g', label='Semi-Supervised')
    plt.xlabel("Etiketli Veri Oranı (%)")
    plt.ylabel("Recall")
    plt.title("Recall vs Etiketli Veri Oranı")
    plt.legend()
    recall_plot_path = './uploads/iris_recall_plot.png'
    plt.savefig(recall_plot_path)
    plt.close()

    return [
        acc_plot_path,
        f1_plot_path,
        precision_plot_path,
        recall_plot_path
    ]
