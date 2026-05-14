"""
BaseKerasTimeSeriesModel
========================
Gemeinsame Logik für LSTMModel/GRUModel/CNNModel: prepare_data, train, evaluate,
predict, plot_results, save_model, run_full_pipeline.

Subklassen überschreiben:
  - MODEL_LABEL  (str, z.B. "LSTM")
  - build_model()  (muss self.model setzen)

Damit sind die ehemaligen ~300 duplizierten Zeilen aus den drei Modell-Dateien
auf eine zentrale Implementierung reduziert. Alle Standalone-Pfade
(`python models/lstm_model.py`) verhalten sich identisch zu vorher.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from data_preparation import DataPreparator, create_sequences


class BaseKerasTimeSeriesModel:
    """Basis-Skeleton für LSTM/GRU/CNN Standalone-Trainings.

    Pipeline-Use (parameter_tuning.FullGridSearchTuner) verwendet diese Klasse
    nicht — sie baut Modelle direkt via build_lstm_model / build_cnn_model /
    build_gru_model. BaseKerasTimeSeriesModel ist ausschließlich für den
    Standalone-Aufruf der drei models/*_model.py-Dateien.
    """

    # Subklassen MUESSEN setzen
    MODEL_LABEL = 'BASE'

    # Subklassen DUERFEN überschreiben
    LOOKBACK_WINDOW = 60
    DROPOUT_RATE = 0.2
    DENSE_UNITS = 32
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    EPOCHS = 100

    def __init__(self, data_path, output_dir):
        self.data_path = data_path
        self.output_dir = output_dir
        self.model = None
        self.preparator = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.history = None
        self.predictions = {}

        os.makedirs(output_dir, exist_ok=True)

    # -- Subklassen überschreiben dies --
    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model().")

    # -- Gemeinsame Pipeline --
    def prepare_data(self):
        print("=" * 70)
        print("STEP 1: Data Preparation (65%-15%-20% split)")
        print("=" * 70)

        self.preparator = DataPreparator(self.data_path)
        train_data, val_data, test_data = self.preparator.load_and_prepare()

        print(f"\nCreating sequences with Lookback Window L={self.LOOKBACK_WINDOW}...")
        self.X_train, self.y_train = create_sequences(train_data, self.LOOKBACK_WINDOW)
        self.X_val,   self.y_val   = create_sequences(val_data,   self.LOOKBACK_WINDOW)
        self.X_test,  self.y_test  = create_sequences(test_data,  self.LOOKBACK_WINDOW)

        print(f"Training sequences: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"Validation sequences: X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"Test sequences: X={self.X_test.shape}, y={self.y_test.shape}")

    def train(self):
        print("\n" + "=" * 70)
        print(f"STEP 3: Train {self.MODEL_LABEL} Model")
        print("=" * 70)

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
            verbose=1,
        )
        print("\nTraining completed!")

    def evaluate(self):
        print("\n" + "=" * 70)
        print("STEP 4: Evaluate Model")
        print("=" * 70)

        train_loss, train_mae = self.model.evaluate(self.X_train, self.y_train, verbose=0)
        val_loss,   val_mae   = self.model.evaluate(self.X_val,   self.y_val,   verbose=0)
        test_loss,  test_mae  = self.model.evaluate(self.X_test,  self.y_test,  verbose=0)
        train_rmse, val_rmse, test_rmse = (np.sqrt(train_loss),
                                           np.sqrt(val_loss),
                                           np.sqrt(test_loss))

        print(f"\nTraining   Loss: {train_loss:.6f}, MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}")
        print(f"Validation Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}")
        print(f"Test       Loss: {test_loss:.6f}, MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}")
        return test_loss, test_mae, test_rmse

    def predict(self):
        print("\n" + "=" * 70)
        print("STEP 5: Generate Predictions")
        print("=" * 70)

        self.predictions['train'] = self.model.predict(self.X_train, verbose=0)
        self.predictions['val']   = self.model.predict(self.X_val,   verbose=0)
        self.predictions['test']  = self.model.predict(self.X_test,  verbose=0)

        for split in ('train', 'val', 'test'):
            self.predictions[f'{split}_original'] = self.preparator.inverse_transform(
                self.predictions[split], split=split, lookback=self.LOOKBACK_WINDOW)

        print("Predictions generated and inverse-transformed!")

    def plot_results(self):
        print("\n" + "=" * 70)
        print("STEP 6: Plot Results")
        print("=" * 70)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history.history['loss'], label='Train Loss')
        axes[0].plot(self.history.history['val_loss'], label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title(f'{self.MODEL_LABEL} Training History - Loss')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(self.history.history['mae'], label='Train MAE')
        axes[1].plot(self.history.history['val_mae'], label='Val MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{self.MODEL_LABEL} Training History - MAE')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(self.output_dir,
                                 f'{self.MODEL_LABEL.lower()}_training_history.png')
        plt.savefig(plot_path)
        print(f"Saved training history plot: {plot_path}")
        plt.close()

    def save_model(self):
        # Keras 3: .keras Format ist Standard. .h5 ist deprecated.
        model_path = os.path.join(self.output_dir,
                                  f'{self.MODEL_LABEL.lower()}_model.keras')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def run_full_pipeline(self):
        self.prepare_data()
        self.build_model()
        self.train()
        self.evaluate()
        self.predict()
        self.plot_results()
        self.save_model()

        print("\n" + "=" * 70)
        print(f"{self.MODEL_LABEL} Pipeline completed!")
        print("=" * 70)
