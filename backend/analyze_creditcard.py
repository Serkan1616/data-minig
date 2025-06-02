import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

def create_autoencoder(input_dim):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dropout(0.3)(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    
    # Decoder
    decoded = Dense(32, activation='relu')(encoded)
    decoded = Dropout(0.3)(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Create models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def preprocess_creditcard(df):
    df = df.copy()
    df.dropna(inplace=True)
    
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(df.drop('Class', axis=1)), 
                    columns=df.drop('Class', axis=1).columns)
    y = df['Class']
    return X, y

def run_experiment(X, y, label_ratio):
    # Ensure we have examples from both classes
    pos_samples = X[y == 1]
    neg_samples = X[y == 0]
    pos_labels = y[y == 1]
    neg_labels = y[y == 0]
    
    # Calculate samples per class
    n_pos = max(1, int(len(pos_samples) * label_ratio))
    n_neg = max(1, int(len(neg_samples) * label_ratio))
    
    # Sample from each class
    pos_indices = np.random.choice(len(pos_samples), n_pos, replace=False)
    neg_indices = np.random.choice(len(neg_samples), n_neg, replace=False)
    
    X_labeled = pd.concat([pos_samples.iloc[pos_indices], neg_samples.iloc[neg_indices]])
    y_labeled = pd.concat([pos_labels.iloc[pos_indices], neg_labels.iloc[neg_indices]])
    
    # Split remaining for unlabeled data
    unlabeled_mask = ~(X.index.isin(X_labeled.index))
    X_unlabeled = X[unlabeled_mask]
    y_unlabeled = y[unlabeled_mask]
    
    X_train_full = pd.concat([X_labeled, X_unlabeled])
    y_train_full = pd.concat([y_labeled, y_unlabeled])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42,
        stratify=y_train_full
    )

    # Supervised model training
    clf_supervised = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    clf_supervised.fit(X_labeled, y_labeled)
    y_pred_supervised = clf_supervised.predict(X_test)

    # Semi-supervised model with Autoencoder
    autoencoder, encoder = create_autoencoder(X.shape[1])
    
    # Train autoencoder on all data
    autoencoder.fit(
        X_train, X_train,
        epochs=50,
        batch_size=256,
        shuffle=True,
        verbose=0
    )
    
    # Get encoded features
    X_train_encoded = encoder.predict(X_train)
    X_test_encoded = encoder.predict(X_test)
    X_labeled_encoded = encoder.predict(X_labeled)
    
    # Train classifier on encoded features
    semi_clf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    semi_clf.fit(X_labeled_encoded, y_labeled)
    y_pred_semi = semi_clf.predict(X_test_encoded)

    return {
        'label_ratio': label_ratio,
        'supervised_f1': f1_score(y_test, y_pred_supervised, zero_division=0),
        'semi_f1': f1_score(y_test, y_pred_semi, zero_division=0),
        'supervised_acc': accuracy_score(y_test, y_pred_supervised),
        'semi_acc': accuracy_score(y_test, y_pred_semi),
        'semi_prec': precision_score(y_test, y_pred_semi, zero_division=0),
        'supervised_prec': precision_score(y_test, y_pred_supervised, zero_division=0),
        'supervised_recall': recall_score(y_test, y_pred_supervised, zero_division=0),
        'semi_recall': recall_score(y_test, y_pred_semi, zero_division=0)
    }

def analyze_creditcard(filepath):
    plt.close('all')
    # Seaborn stili yerine matplotlib'in kendi modern stilini kullanalım
    plt.style.use('bmh')  # Alternatif olarak: 'ggplot', 'fivethirtyeight', etc.
    
    df = pd.read_csv(filepath)
    X, y = preprocess_creditcard(df)

    # Plot dosyalarını temizle
    for plot_file in ['credit_accuracy_plot.png', 'credit_f1_plot.png', 
                      'credit_precision_plot.png', 'credit_recall_plot.png']:
        try:
            os.remove(os.path.join('./uploads', plot_file))
        except:
            pass

    ratios = [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0]
    results = []

    for ratio in ratios:
        result = run_experiment(X, y, ratio)
        results.append(result)
        print(f"Ratio: {ratio}, Supervised F1: {result['supervised_f1']:.3f}, Semi F1: {result['semi_f1']:.3f}")

    results_df = pd.DataFrame(results)

    # Plot functions
    def create_plot(metric_name, y_sup, y_semi, title):
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['label_ratio']*100, y_sup, 'o-', color='#1f77b4', 
                linewidth=2, markersize=8, label='Supervised')
        plt.plot(results_df['label_ratio']*100, y_semi, 'o-', color='#2ca02c', 
                linewidth=2, markersize=8, label='Semi-supervised')
        plt.grid(True, alpha=0.3)
        plt.xlabel("Labeled Data Ratio (%)", fontsize=12)
        plt.ylabel(f"{metric_name}", fontsize=12)
        plt.title(f"{title}", fontsize=14, pad=20)
        plt.legend(fontsize=10, framealpha=0.8)
        plt.tight_layout()
        # Stil geliştirmeleri
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        return plt

    # Accuracy Plot
    acc_plot = create_plot("Accuracy Score", 
                          results_df['supervised_acc'], 
                          results_df['semi_acc'],
                          "Credit Card Fraud Detection: Accuracy Comparison")
    acc_plot_path = './uploads/credit_accuracy_plot.png'
    acc_plot.savefig(acc_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # F1 Score Plot
    f1_plot = create_plot("F1 Score", 
                         results_df['supervised_f1'], 
                         results_df['semi_f1'],
                         "Credit Card Fraud Detection: F1 Score Comparison")
    f1_plot_path = './uploads/credit_f1_plot.png'
    f1_plot.savefig(f1_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Precision Plot
    prec_plot = create_plot("Precision Score", 
                           results_df['supervised_prec'], 
                           results_df['semi_prec'],
                           "Credit Card Fraud Detection: Precision Comparison")
    prec_plot_path = './uploads/credit_precision_plot.png'
    prec_plot.savefig(prec_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Recall Plot
    recall_plot = create_plot("Recall Score", 
                            results_df['supervised_recall'], 
                            results_df['semi_recall'],
                            "Credit Card Fraud Detection: Recall Comparison")
    recall_plot_path = './uploads/credit_recall_plot.png'
    recall_plot.savefig(recall_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print("\nMetrics Summary:")
    print(results_df.round(3))

    return [acc_plot_path, f1_plot_path, prec_plot_path, recall_plot_path]
