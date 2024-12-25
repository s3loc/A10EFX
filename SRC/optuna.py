import optuna
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Embedding, LSTM
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, confusion_matrix, matthews_corrcoef, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import joblib
import numpy as np
import multiprocessing
from datetime import datetime
from ray import tune
from keras_tuner import RandomSearch
from sklearn.utils import resample

# Dağıtık eğitim için TensorFlow MirroredStrategy
def get_strategy():
    try:
        strategy = tf.distribute.MirroredStrategy()
    except Exception as e:
        strategy = tf.distribute.get_strategy()
    return strategy

# GAN tabanlı veri artırma
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Model

def build_gan(data_shape):
    generator = Sequential([
        Dense(128, activation=LeakyReLU(0.2), input_dim=data_shape),
        Dense(256, activation=LeakyReLU(0.2)),
        Dense(data_shape, activation='tanh')
    ])

    discriminator = Sequential([
        Dense(256, activation=LeakyReLU(0.2), input_dim=data_shape),
        Dense(128, activation=LeakyReLU(0.2)),
        Dense(1, activation='sigmoid')
    ])

    discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
    z = tf.keras.Input(shape=(data_shape,))
    generated_data = generator(z)
    discriminator.trainable = False
    validity = discriminator(generated_data)
    combined = Model(z, validity)
    combined.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

    return generator, discriminator, combined

# Hiperparametre optimizasyonu ve Neural Architecture Search
def create_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    for i in range(hp.Int('num_layers', 1, 5)):
        model.add(Dense(hp.Int(f'units_{i}', min_value=32, max_value=256, step=32), activation='relu'))
        model.add(Dropout(hp.Float(f'dropout_{i}', 0.1, 0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-5, 1e-2, sampling='log')),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Veri artırma ve işleme
smote = SMOTE()
def preprocess_data(X, y):
    X, y = smote.fit_resample(X, y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    return X, y

# Eğitim ve optimizasyon süreci
strategy = get_strategy()
def train_model(X, y):
    X, y = preprocess_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tuner = RandomSearch(
        create_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_logs',
        project_name='nas_project')

    tuner.search(X_train, y_train, epochs=100, validation_split=0.2,
                 callbacks=[EarlyStopping(monitor='val_loss', patience=5)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32,
              callbacks=[TensorBoard(log_dir=f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'),
                         EarlyStopping(monitor='val_loss', patience=5)])

    preds = model.predict(X_test)
    preds_binary = (preds > 0.5).astype('int32')

    metrics = {
        'accuracy': accuracy_score(y_test, preds_binary),
        'f1_score': f1_score(y_test, preds_binary, average='weighted'),
        'precision': precision_score(y_test, preds_binary, average='weighted'),
        'recall': recall_score(y_test, preds_binary, average='weighted'),
        'roc_auc': roc_auc_score(y_test, preds),
        'mcc': matthews_corrcoef(y_test, preds_binary),
        'log_loss': log_loss(y_test, preds)
    }
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    return model
