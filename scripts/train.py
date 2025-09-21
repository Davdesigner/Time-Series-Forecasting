from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, X_train, y_train, model_name="Model", 
                epochs=50, batch_size=32, val_split=0.2, patience=10, verbose=1):
    """
    Trains the given Keras model with early stopping.

    Args:
        model: Compiled Keras model
        X_train (np.ndarray): Scaled input features
        y_train (np.ndarray or pd.Series): Target values
        model_name (str): Name for logs
        epochs (int): Max training epochs
        batch_size (int): Batch size
        val_split (float): Fraction of training data used for validation
        patience (int): Early stopping patience
        verbose (int): Verbosity level (1 for full logs, 0 for silent)

    Returns:
        model: The trained model
        history: Keras History object
    """
    print(f"🚀 Starting training: {model_name}")

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=verbose
    )

    print(f"✅ Training completed: {model_name}")
    return model, history
