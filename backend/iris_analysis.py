# iris_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.semi_supervised import SelfTrainingClassifier, LabelPropagation, LabelSpreading

def analyze_iris(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(['Id', 'Species'], axis=1)
    y = LabelEncoder().fit_transform(data['Species'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

    models = {
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=200),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    accuracies = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[name] = accuracy_score(y_test, y_pred)

    os.makedirs('./static/plots', exist_ok=True)

    plt.figure(figsize=(10,6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette='husl')
    plt.ylabel("Accuracy")
    plt.title("Supervised Model Performans Karşılaştırması")
    plt.xticks(rotation=45)
    plt.ylim(0.8, 1.0)
    supervised_plot_path = './static/plots/supervised.png'
    plt.savefig(supervised_plot_path)
    plt.close()

    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    y_pred_best = best_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_best)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{best_model_name} - Confusion Matrix")
    cm_supervised_path = './static/plots/supervised_cm.png'
    plt.savefig(cm_supervised_path)
    plt.close()

    y_train_semi = np.copy(y_train)
    rng = np.random.RandomState(42)
    n_labeled = int(0.3 * len(y_train_semi))
    y_train_semi[rng.choice(len(y_train_semi), len(y_train_semi) - n_labeled, replace=False)] = -1

    semi_models = {
        "Self-Training (KNN)": SelfTrainingClassifier(KNeighborsClassifier(n_neighbors=5)),
        "Label Propagation": LabelPropagation(),
        "Label Spreading": LabelSpreading()
    }

    semi_accuracies = {}
    for name, model in semi_models.items():
        model.fit(X_train, y_train_semi)
        y_pred = model.predict(X_test)
        semi_accuracies[name] = accuracy_score(y_test, y_pred)

    plt.figure(figsize=(10,6))
    sns.barplot(x=list(semi_accuracies.keys()), y=list(semi_accuracies.values()), palette='pastel')
    plt.ylabel("Accuracy")
    plt.title("Semi-Supervised Model Performansları")
    plt.xticks(rotation=45)
    plt.ylim(0.8, 1.0)
    semi_plot_path = './static/plots/semi.png'
    plt.savefig(semi_plot_path)
    plt.close()

    best_semi_model_name = max(semi_accuracies, key=semi_accuracies.get)
    best_semi_model = semi_models[best_semi_model_name]
    y_pred_semi = best_semi_model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred_semi)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Purples)
    plt.title(f"{best_semi_model_name} - Confusion Matrix")
    cm_semi_path = './static/plots/semi_cm.png'
    plt.savefig(cm_semi_path)
    plt.close()

    final_comparison = {
        f"Supervised - {best_model_name}": accuracies[best_model_name],
        f"Semi-Supervised - {best_semi_model_name}": semi_accuracies[best_semi_model_name]
    }

    plt.figure(figsize=(8,5))
    sns.barplot(x=list(final_comparison.keys()), y=list(final_comparison.values()), palette='Set2')
    plt.ylabel("Accuracy")
    plt.title("Final Karşılaştırma")
    plt.ylim(0.8, 1.0)
    final_path = './static/plots/final.png'
    plt.savefig(final_path)
    plt.close()

    return [
        supervised_plot_path,
        cm_supervised_path,
        semi_plot_path,
        cm_semi_path,
        final_path
    ]
