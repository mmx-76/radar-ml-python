"""Train a simple MLP classifier for radar range-profile classification."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier


# File paths for the training, validation, and test datasets
TRAIN_FEATURES_PATH = "data/training_vectors/X_train.npy"
TRAIN_LABELS_PATH = "data/training_vectors/y_train.npy"
VAL_FEATURES_PATH = "data/training_vectors/X_val.npy"
VAL_LABELS_PATH = "data/training_vectors/y_val.npy"
TEST_FEATURES_PATH = "data/training_vectors/X_test.npy"
TEST_LABELS_PATH = "data/training_vectors/y_test.npy"

# Output file paths for the trained model parameters
HIDDEN_WEIGHTS_PATH = "ml_weights_hidden.npy"
HIDDEN_BIASES_PATH = "ml_biases_hidden.npy"
OUTPUT_WEIGHTS_PATH = "ml_weights_output.npy"
OUTPUT_BIASES_PATH = "ml_biases_output.npy"


def load_datasets():
    """Load training, validation, and test datasets from .npy files."""
    X_train = np.load(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH)
    X_val = np.load(VAL_FEATURES_PATH)
    y_val = np.load(VAL_LABELS_PATH)
    X_test = np.load(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH)
    return X_train, y_train, X_val, y_val, X_test, y_test


def build_model(random_state: int = 42) -> MLPClassifier:
    """Create and return an MLPClassifier with the required architecture."""
    return MLPClassifier(
        hidden_layer_sizes=(6,),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=random_state,
    )


def train_and_evaluate():
    """Train the model, evaluate it, print metrics, and save weights."""
    # Load datasets
    X_train, y_train, X_val, y_val, X_test, y_test = load_datasets()

    # Initialize and train the MLP classifier
    model = build_model()
    model.fit(X_train, y_train)

    # Compute predictions for each split
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, train_preds)
    val_accuracy = accuracy_score(y_val, val_preds)
    test_accuracy = accuracy_score(y_test, test_preds)

    # Display performance metrics
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Confusion matrix and classification report on the test set
    test_confusion = confusion_matrix(y_test, test_preds)
    print("Confusion matrix:")
    print(test_confusion)

    test_report = classification_report(y_test, test_preds)
    print("Classification report:")
    print(test_report)

    # Save trained model parameters
    hidden_weights, output_weights = model.coefs_
    hidden_biases, output_biases = model.intercepts_

    np.save(HIDDEN_WEIGHTS_PATH, hidden_weights)
    np.save(HIDDEN_BIASES_PATH, hidden_biases)
    np.save(OUTPUT_WEIGHTS_PATH, output_weights)
    np.save(OUTPUT_BIASES_PATH, output_biases)


if __name__ == "__main__":
    train_and_evaluate()
