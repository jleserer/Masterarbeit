"""
Comprehensive Test Suite for LSTM and CNN Models
================================================

Tests:
  1. Data preparation (loading, splitting, shape validation)
  2. Sequence creation (Loopback-Window)
  3. Model building (LSTM & CNN)
  4. Model predictions (shapes and ranges)
  5. Inverse transform
  6. Training capability
"""

import os
import sys
import unittest
import numpy as np
import pandas as pd
from data_preparation import DataPreparator, create_sequences
from lstm_model import LSTMModel
from cnn_model import CNNModel


class TestDataPreparation(unittest.TestCase):
    """Test data loading and preprocessing."""
    
    @classmethod
    def setUpClass(cls):
        """Load data once for all tests."""
        cls.data_path = os.path.join('..', 'stockData', 'normalizedData', 'SP500_historical_data.csv')
        cls.preparator = DataPreparator(cls.data_path)
    
    def test_file_exists(self):
        """Test data file exists."""
        self.assertTrue(os.path.exists(self.data_path), f"Data file not found: {self.data_path}")
    
    def test_data_loading(self):
        """Test CSV loading."""
        df = pd.read_csv(self.data_path)
        self.assertGreater(len(df), 0, "DataFrame is empty")
        self.assertIn('Close', df.columns, "Close column not found")
    
    def test_load_and_prepare(self):
        """Test data preparation with 65%-15%-20% split."""
        train, val, test = self.preparator.load_and_prepare()
        
        # Check shapes
        self.assertEqual(train.ndim, 2, "Train data should be 2D")
        self.assertEqual(val.ndim, 2, "Val data should be 2D")
        self.assertEqual(test.ndim, 2, "Test data should be 2D")
        
        # Check split percentages
        total = len(train) + len(val) + len(test)
        train_pct = len(train) / total
        val_pct = len(val) / total
        test_pct = len(test) / total
        
        self.assertAlmostEqual(train_pct, 0.65, places=2, msg="Train split not ~65%")
        self.assertAlmostEqual(val_pct, 0.15, places=2, msg="Val split not ~15%")
        self.assertAlmostEqual(test_pct, 0.20, places=2, msg="Test split not ~20%")
    
    def test_single_feature_output(self):
        """Test that only Close is used (single feature)."""
        train, val, test = self.preparator.load_and_prepare()
        
        # Should be (n_samples, 1) for single feature
        self.assertEqual(train.shape[1], 1, "Train should have 1 feature (Close only)")
        self.assertEqual(val.shape[1], 1, "Val should have 1 feature (Close only)")
        self.assertEqual(test.shape[1], 1, "Test should have 1 feature (Close only)")
    
    def test_normalization_range(self):
        """Test data is normalized to [0, 1]."""
        train, val, test = self.preparator.load_and_prepare()
        
        all_data = np.concatenate([train, val, test])
        self.assertTrue(np.all(all_data >= 0), "Data contains values < 0")
        self.assertTrue(np.all(all_data <= 1), "Data contains values > 1")
    
    def test_no_nan_values(self):
        """Test no NaN values after cleaning."""
        train, val, test = self.preparator.load_and_prepare()
        
        self.assertFalse(np.any(np.isnan(train)), "Train data contains NaN")
        self.assertFalse(np.any(np.isnan(val)), "Val data contains NaN")
        self.assertFalse(np.any(np.isnan(test)), "Test data contains NaN")


class TestSequenceCreation(unittest.TestCase):
    """Test Loopback-Window sequence generation."""
    
    @classmethod
    def setUpClass(cls):
        """Prepare data for sequence tests."""
        cls.data_path = os.path.join('..', 'stockData', 'normalizedData', 'SP500_historical_data.csv')
        cls.preparator = DataPreparator(cls.data_path)
        cls.train, cls.val, cls.test = cls.preparator.load_and_prepare()
    
    def test_sequence_shapes(self):
        """Test sequence generation creates correct shapes."""
        L = 30
        X_train, y_train = create_sequences(self.train, L)
        X_val, y_val = create_sequences(self.val, L)
        X_test, y_test = create_sequences(self.test, L)
        
        # X should be (n_sequences, L, 1)
        self.assertEqual(X_train.shape[1], L, f"X_train second dim should be {L}")
        self.assertEqual(X_train.shape[2], 1, "X_train third dim should be 1 (single feature)")
        
        # y should be (n_sequences, 1)
        self.assertEqual(y_train.shape[1], 1, "y_train second dim should be 1")
    
    def test_sequence_count(self):
        """Test correct number of sequences created."""
        L = 30
        data_len = len(self.train)
        X_train, y_train = create_sequences(self.train, L)
        
        # Should create (data_len - L) sequences
        expected_count = data_len - L
        self.assertEqual(len(X_train), expected_count, 
                        f"Expected {expected_count} sequences, got {len(X_train)}")
    
    def test_sequence_continuity(self):
        """Test sequences are continuous from data."""
        L = 30
        X_train, y_train = create_sequences(self.train, L)
        
        # First sequence should match first L elements
        np.testing.assert_array_almost_equal(X_train[0], self.train[:L], 
                                            decimal=5, err_msg="First sequence mismatch")
        
        # Target should be the next element
        np.testing.assert_array_almost_equal(y_train[0], self.train[L], 
                                            decimal=5, err_msg="First target mismatch")
    
    def test_different_window_sizes(self):
        """Test sequence creation with different window sizes."""
        for L in [10, 20, 30, 60]:
            X, y = create_sequences(self.train, L)
            self.assertEqual(X.shape[1], L, f"Window size {L} not respected")
            self.assertEqual(len(X), len(self.train) - L, f"Wrong sequence count for L={L}")


class TestLSTMModel(unittest.TestCase):
    """Test LSTM model."""
    
    @classmethod
    def setUpClass(cls):
        """Prepare LSTM model."""
        cls.data_path = os.path.join('..', 'stockData', 'normalizedData', 'SP500_historical_data.csv')
        
        # Create quick LSTM for testing
        class QuickLSTM(LSTMModel):
            EPOCHS = 1  # Single epoch for testing
            LOOKBACK_WINDOW = 30
        
        cls.lstm = QuickLSTM(cls.data_path, output_dir='test_lstm_output')
        cls.lstm.prepare_data()
        cls.lstm.build_model()
    
    def test_model_exists(self):
        """Test model is built."""
        self.assertIsNotNone(self.lstm.model, "LSTM model not built")
    
    def test_model_output_shape(self):
        """Test model output layer is single feature."""
        output_shape = self.lstm.model.output_shape
        # Should be (None, 1) for single output
        self.assertEqual(output_shape[-1], 1, f"Output should be 1 feature, got {output_shape[-1]}")
    
    def test_model_input_shape(self):
        """Test model input shape matches sequences."""
        input_shape = self.lstm.model.input_shape
        # Should be (None, 30, 1) for LOOKBACK_WINDOW=30, single feature
        self.assertEqual(input_shape[1], 30, f"Input time steps should be 30, got {input_shape[1]}")
        self.assertEqual(input_shape[2], 1, f"Input features should be 1, got {input_shape[2]}")
    
    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        predictions = self.lstm.model.predict(self.lstm.X_test[:10], verbose=0)
        
        # Should be (n_samples, 1)
        self.assertEqual(predictions.shape[0], 10, "Prediction batch size mismatch")
        self.assertEqual(predictions.shape[1], 1, "Prediction should output 1 value")
    
    def test_prediction_range(self):
        """Test predictions are in normalized range [0, 1]."""
        predictions = self.lstm.model.predict(self.lstm.X_test[:100], verbose=0)
        
        # Should be roughly in [0, 1] (allow small overshoot due to extrapolation)
        self.assertTrue(np.all(predictions >= -0.1), "Predictions below normalized range")
        self.assertTrue(np.all(predictions <= 1.1), "Predictions above normalized range")
    
    def test_model_trainable(self):
        """Test model can be trained."""
        # Try training for 1 epoch
        history = self.lstm.model.fit(
            self.lstm.X_train[:100], 
            self.lstm.y_train[:100],
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        self.assertIn('loss', history.history, "Loss not in training history")
        self.assertGreater(history.history['loss'][0], 0, "Loss should be positive")
    
    def test_inverse_transform(self):
        """Test inverse transform works."""
        normalized_pred = np.array([[0.5], [0.7], [0.3]])
        original_pred = self.lstm.preparator.inverse_transform(normalized_pred)
        
        # Should be back in original scale
        self.assertEqual(original_pred.shape, normalized_pred.shape, "Shape mismatch after inverse")
        self.assertGreater(original_pred[0, 0], 1, "Inverse transform may not be working")


class TestCNNModel(unittest.TestCase):
    """Test CNN model."""
    
    @classmethod
    def setUpClass(cls):
        """Prepare CNN model."""
        cls.data_path = os.path.join('..', 'stockData', 'normalizedData', 'SP500_historical_data.csv')
        
        # Create quick CNN for testing
        class QuickCNN(CNNModel):
            EPOCHS = 1  # Single epoch for testing
            LOOKBACK_WINDOW = 30
        
        cls.cnn = QuickCNN(cls.data_path, output_dir='test_cnn_output')
        cls.cnn.prepare_data()
        cls.cnn.build_model()
    
    def test_model_exists(self):
        """Test model is built."""
        self.assertIsNotNone(self.cnn.model, "CNN model not built")
    
    def test_model_output_shape(self):
        """Test model output layer is single feature."""
        output_shape = self.cnn.model.output_shape
        # Should be (None, 1) for single output
        self.assertEqual(output_shape[-1], 1, f"Output should be 1 feature, got {output_shape[-1]}")
    
    def test_model_input_shape(self):
        """Test model input shape matches sequences."""
        input_shape = self.cnn.model.input_shape
        # Should be (None, 30, 1) for LOOKBACK_WINDOW=30, single feature
        self.assertEqual(input_shape[1], 30, f"Input time steps should be 30, got {input_shape[1]}")
        self.assertEqual(input_shape[2], 1, f"Input features should be 1, got {input_shape[2]}")
    
    def test_model_prediction_shape(self):
        """Test prediction output shape."""
        predictions = self.cnn.model.predict(self.cnn.X_test[:10], verbose=0)
        
        # Should be (n_samples, 1)
        self.assertEqual(predictions.shape[0], 10, "Prediction batch size mismatch")
        self.assertEqual(predictions.shape[1], 1, "Prediction should output 1 value")
    
    def test_prediction_range(self):
        """Test predictions are in normalized range [0, 1]."""
        predictions = self.cnn.model.predict(self.cnn.X_test[:100], verbose=0)
        
        # Should be roughly in [0, 1] (allow small overshoot due to extrapolation)
        self.assertTrue(np.all(predictions >= -0.1), "Predictions below normalized range")
        self.assertTrue(np.all(predictions <= 1.1), "Predictions above normalized range")
    
    def test_model_trainable(self):
        """Test model can be trained."""
        # Try training for 1 epoch
        history = self.cnn.model.fit(
            self.cnn.X_train[:100], 
            self.cnn.y_train[:100],
            epochs=1,
            batch_size=32,
            verbose=0
        )
        
        self.assertIn('loss', history.history, "Loss not in training history")
        self.assertGreater(history.history['loss'][0], 0, "Loss should be positive")
    
    def test_inverse_transform(self):
        """Test inverse transform works."""
        normalized_pred = np.array([[0.5], [0.7], [0.3]])
        original_pred = self.cnn.preparator.inverse_transform(normalized_pred)
        
        # Should be back in original scale
        self.assertEqual(original_pred.shape, normalized_pred.shape, "Shape mismatch after inverse")
        self.assertGreater(original_pred[0, 0], 1, "Inverse transform may not be working")


class TestModelComparison(unittest.TestCase):
    """Test model comparison between LSTM and CNN."""
    
    @classmethod
    def setUpClass(cls):
        """Build both models."""
        cls.data_path = os.path.join('..', 'stockData', 'normalizedData', 'SP500_historical_data.csv')
        
        class QuickLSTM(LSTMModel):
            EPOCHS = 1
            LOOKBACK_WINDOW = 30
        
        class QuickCNN(CNNModel):
            EPOCHS = 1
            LOOKBACK_WINDOW = 30
        
        cls.lstm = QuickLSTM(cls.data_path, output_dir='test_lstm')
        cls.cnn = QuickCNN(cls.data_path, output_dir='test_cnn')
        
        cls.lstm.prepare_data()
        cls.cnn.prepare_data()
        
        cls.lstm.build_model()
        cls.cnn.build_model()
    
    def test_both_models_same_input_shape(self):
        """Test both models accept same input shape."""
        lstm_input = self.lstm.model.input_shape
        cnn_input = self.cnn.model.input_shape
        
        self.assertEqual(lstm_input, cnn_input, 
                        f"Input shapes differ: LSTM {lstm_input} vs CNN {cnn_input}")
    
    def test_both_models_same_output_shape(self):
        """Test both models produce same output shape."""
        lstm_output = self.lstm.model.output_shape
        cnn_output = self.cnn.model.output_shape
        
        self.assertEqual(lstm_output, cnn_output,
                        f"Output shapes differ: LSTM {lstm_output} vs CNN {cnn_output}")
    
    def test_both_models_same_data_split(self):
        """Test both models use same data split."""
        self.assertEqual(len(self.lstm.X_train), len(self.cnn.X_train),
                        "LSTM and CNN training set sizes differ")
        self.assertEqual(len(self.lstm.X_val), len(self.cnn.X_val),
                        "LSTM and CNN validation set sizes differ")
        self.assertEqual(len(self.lstm.X_test), len(self.cnn.X_test),
                        "LSTM and CNN test set sizes differ")
    
    def test_models_produce_different_predictions(self):
        """Test models produce different predictions (shouldn't be identical)."""
        test_sample = self.lstm.X_test[:10]
        
        lstm_pred = self.lstm.model.predict(test_sample, verbose=0)
        cnn_pred = self.cnn.model.predict(test_sample, verbose=0)
        
        # Models should differ (very unlikely to be identical)
        self.assertFalse(np.allclose(lstm_pred, cnn_pred, atol=1e-3),
                        "LSTM and CNN predictions are suspiciously identical")


def run_tests_with_summary():
    """Run all tests and print summary."""
    import sys
    # Fix encoding for Windows PowerShell
    if sys.stdout.encoding != 'utf-8':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("\n" + "="*80)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*80 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPreparation))
    suite.addTests(loader.loadTestsFromTestCase(TestSequenceCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestLSTMModel))
    suite.addTests(loader.loadTestsFromTestCase(TestCNNModel))
    suite.addTests(loader.loadTestsFromTestCase(TestModelComparison))
    
    # Run tests with verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n[PASS] ALL TESTS PASSED!")
    else:
        print("\n[FAIL] SOME TESTS FAILED")
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    print("="*80 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests_with_summary()
    sys.exit(0 if success else 1)
