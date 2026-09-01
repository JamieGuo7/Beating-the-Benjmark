from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def create_callbacks(ticker, MODELS_DIR):
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=0
        ),
        ModelCheckpoint(
            filepath=f'{MODELS_DIR}/lstm_{ticker}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
    ]

    return callbacks