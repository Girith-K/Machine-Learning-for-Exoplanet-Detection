import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix, 
                           classification_report, roc_curve)
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
np.random.seed(42)
tf.random.set_seed(42)
print("Loading exoplanet dataset...")
try:
    possible_names = ['exoplanets.csv']
    data = None
    
    for name in possible_names:
        try:
            if name.endswith('.csv'):
                data = pd.read_csv(name)
            else:
                data = pd.read_excel(name)
            print(f"Successfully loaded {name}")
            break
        except FileNotFoundError:
            continue
    
    if data is None:
        print("Dataset file not found. Please ensure the file is in the project directory.")
        print("Expected name: exoplanets.csv")
        exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()
print(f"Dataset shape: {data.shape}")
print(f"Columns ({len(data.columns)}): {list(data.columns)[:10]}...")  # Show first 10 columns
print(f"\nFirst few rows:")
print(data.head())
print("\nData types of columns:")
print(data.dtypes.value_counts())
target_column = None
possible_target_names = ['koi_disposition', 'disposition', 'label', 'target', 'class']
for col in possible_target_names:
    if col in data.columns:
        target_column = col
        break
if target_column is None:
    print(f"\nWarning: Could not identify target column. Using 'koi_disposition' as default.")
    target_column = 'koi_disposition'
print(f"\nTarget column: {target_column}")
print(f"Target distribution:\n{data[target_column].value_counts()}")
def create_binary_target(disposition):
    """Convert disposition to binary classification"""
    if pd.isna(disposition):
        return np.nan
    disposition = str(disposition).upper()
    if 'CONFIRMED' in disposition:
        return 1
    else:
        return 0
data['target'] = data[target_column].apply(create_binary_target)
data = data.dropna(subset=['target'])
data['target'] = data['target'].astype(int)
print(f"\nBinary target distribution:")
print(f"0 (Not Confirmed): {sum(data['target'] == 0)}")
print(f"1 (Confirmed):     {sum(data['target'] == 1)}")
numerical_columns = data.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numerical_columns:
    numerical_columns.remove('target')
columns_to_remove = ['kepid', 'kepoi_name', 'kepler_name', 'rowid', 'id']
for col in columns_to_remove:
    if col in numerical_columns:
        numerical_columns.remove(col)
print(f"\nNumber of numerical features: {len(numerical_columns)}")
print(f"Features (first 20): {numerical_columns[:20]}")
X = data[numerical_columns].values
y = data['target'].values
print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print("\nHandling missing values...")
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
print(f"NaN values in features: {np.isnan(X).sum()}")
print(f"Inf values in features: {np.isinf(X).sum()}")
if np.isinf(X).sum() > 0:
    print("Replacing infinite values...")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Test set: {X_test.shape}, {y_test.shape}")
print("\nApplying oversampling to handle class imbalance...")
oversampler = RandomOverSampler(sampling_strategy=0.5, random_state=42)
X_train_balanced, y_train_balanced = oversampler.fit_resample(X_train, y_train)
print(f"After oversampling:")
print(f"Class 0 count: {sum(y_train_balanced == 0)}")
print(f"Class 1 count: {sum(y_train_balanced == 1)}")
def create_dnn_model(input_dim):
    """Create a DNN model for exoplanet detection"""
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(1, activation='sigmoid')
    ])
    return model
print("\nCreating Deep Neural Network model...")
input_dim = X_train.shape[1]
model = create_dnn_model(input_dim)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
             tf.keras.metrics.Recall(name='recall')]
)
print("\nModel architecture:")
model.summary()
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, monitor='val_loss')
]
print("\nTraining the model...")
history = model.fit(
    X_train_balanced, y_train_balanced,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks,
    verbose=1
)
print("\nMaking predictions...")
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()
y_pred_prob = y_pred_prob.flatten()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
try:
    roc_auc = roc_auc_score(y_test, y_pred_prob)
except:
    roc_auc = 0.0
    print("Warning: Could not calculate ROC AUC (possibly due to single class in predictions)")
print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")
print("="*50)
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=["Not Confirmed", "Confirmed Exoplanet"]))
print("\nAnalyzing feature importance...")
feature_importance_scores = []
for i in range(min(10, len(numerical_columns))):  
    X_test_permuted = X_test.copy()
    np.random.shuffle(X_test_permuted[:, i])
    y_pred_permuted = (model.predict(X_test_permuted) > 0.5).astype(int).flatten()
    accuracy_permuted = accuracy_score(y_test, y_pred_permuted)
    importance = accuracy - accuracy_permuted
    feature_importance_scores.append((numerical_columns[i], importance))
feature_importance_scores.sort(key=lambda x: x[1], reverse=True)
print("\nTop 10 Most Important Features:")
for i, (feature, importance) in enumerate(feature_importance_scores[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Exoplanet Detection Model Results', fontsize=16, fontweight='bold')
axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0, 0].set_title('Model Accuracy Over Epochs')
axes[0, 0].set_xlabel('Epochs')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[0, 1].set_title('Model Loss Over Epochs')
axes[0, 1].set_xlabel('Epochs')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
if roc_auc > 0:
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[0, 2].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0, 2].set_title('ROC Curve')
    axes[0, 2].set_xlabel('False Positive Rate')
    axes[0, 2].set_ylabel('True Positive Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
else:
    axes[0, 2].text(0.5, 0.5, 'ROC Curve not available', ha='center', va='center')
    axes[0, 2].set_title('ROC Curve')
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
            xticklabels=['Not Confirmed', 'Confirmed'],
            yticklabels=['Not Confirmed', 'Confirmed'])
axes[1, 0].set_title('Confusion Matrix')
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 1].hist(y_pred_prob[y_test == 0], bins=30, alpha=0.7, 
                label='Not Confirmed', density=True, color='coral')
axes[1, 1].hist(y_pred_prob[y_test == 1], bins=30, alpha=0.7, 
                label='Confirmed', density=True, color='skyblue')
axes[1, 1].set_title('Prediction Probability Distribution')
axes[1, 1].set_xlabel('Predicted Probability')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
metrics_values = [accuracy, precision, recall, f1, roc_auc]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
bars = axes[1, 2].bar(metrics_names, metrics_values, color=colors)
axes[1, 2].set_title('Model Performance Metrics')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_ylim(0, 1.1)
for bar, value in zip(bars, metrics_values):
    axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
axes[1, 2].grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()
print("\nCreating feature distribution plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Feature Distributions by Target Class', fontsize=16, fontweight='bold')
features_to_plot = numerical_columns[:6] if len(numerical_columns) >= 6 else numerical_columns
for idx, (ax, feature) in enumerate(zip(axes.flat, features_to_plot)):
    if feature in data.columns:
        feature_data = pd.DataFrame({
            feature: data[feature],
            'Target': data['target'].map({0: 'Not Confirmed', 1: 'Confirmed'})
        })
        sns.violinplot(data=feature_data, x='Target', y=feature, ax=ax)
        ax.set_title(f'Distribution of {feature}')
        ax.grid(True, alpha=0.3)
for idx in range(len(features_to_plot), 6):
    axes.flat[idx].set_visible(False)
plt.tight_layout()
plt.show()
print(f"\nModel training completed!")
print(f"Final test accuracy: {accuracy:.4f}")
print(f"Total parameters: {model.count_params():,}")
save_model = input("\nWould you like to save the trained model? (y/n): ").lower()
if save_model == 'y':
    model.save('exoplanet_detection_model.h5')
    print("Model saved as 'exoplanet_detection_model.h5'")
    import joblib
    joblib.dump(scaler, 'exoplanet_scaler.pkl')
    print("Scaler saved as 'exoplanet_scaler.pkl'")
    with open('feature_names.txt', 'w') as f:
        for feature in numerical_columns:
            f.write(f"{feature}\n")

    print("Feature names saved as 'feature_names.txt'")
