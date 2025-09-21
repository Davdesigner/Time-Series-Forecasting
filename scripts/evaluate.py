import numpy as np
import matplotlib.pyplot as plt

def evaluate_model_training(model, history, X_train, y_train, model_name="Model"):
    """
    Evaluate and visualize the training loss of a model.

    Args:
        model: Trained Keras model
        history: Training history object from model.fit()
        X_train (np.ndarray): Input features used for training
        y_train (pd.Series or np.ndarray): True target values
        model_name (str): Name used for labeling plots and printouts

    Returns:
        float: Final RMSE on training set
    """

    # Predict on training data
    predictions = model.predict(X_train).flatten()
    y_true = y_train.values if hasattr(y_train, 'values') else y_train

    # Compute RMSE
    rmse = np.sqrt(np.mean((y_true - predictions) ** 2))

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.axhline(y=rmse**2, color='blue', linestyle='--', label='Final Train MSE')
    plt.title(f'{model_name} - Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f" Final Training RMSE for {model_name}: {rmse:.4f}")
    return rmse
