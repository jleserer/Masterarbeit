"""
Quick Test of LSTM and CNN Models
"""
import os
import sys

print("Testing data preparation...")
from data_preparation import DataPreparator, create_sequences

data_path = os.path.join('..', 'stockData', 'preprocessedData', 'SP500_historical_data.csv')
preparator = DataPreparator(data_path)
train, val, test = preparator.load_and_prepare()
print("✓ Data preparation successful")

print("\nTesting sequence creation...")
X_train, y_train = create_sequences(train, 30)
print(f"✓ Sequences created: X={X_train.shape}, y={y_train.shape}")

print("\nTesting LSTM model...")
from lstm_model import LSTMModel

class QuickLSTM(LSTMModel):
    EPOCHS = 2

lstm = QuickLSTM(data_path, output_dir='lstm_quick_test')
lstm.prepare_data()
lstm.build_model()
print("✓ LSTM model built successfully")

print("\nTesting CNN model...")
from cnn_model import CNNModel

class QuickCNN(CNNModel):
    EPOCHS = 2

cnn = CNNModel(data_path, output_dir='cnn_quick_test')
cnn.prepare_data()
cnn.build_model()
print("✓ CNN model built successfully")

print("\n" + "="*70)
print("✓ ALL TESTS PASSED!")
print("="*70)
print("\nModels are ready for training. Run:")
print("  python lstm_model.py   # Full LSTM training")
print("  python cnn_model.py    # Full CNN training")
print("  python model_comparison.py  # Train both and compare")
