import tensorflow as tf
from tensorflow import keras

def create_dnn_model(input_shape, num_classes):
    """
    Creates a DNN model for multiclass intrusion detection.
    
    Args:
        input_shape (int): The number of input features (columns).
        num_classes (int): The number of unique attack labels in the data.
    """
    
    model = keras.Sequential([
        # Input Layer
        keras.Input(shape=(input_shape,)),
        
        # Hidden Layers - Deepened & Enhanced
        keras.layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        
        # Output Layer
        keras.layers.Dense(num_classes, activation='softmax'), 
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(),
        # Use sparse_categorical_crossentropy because targets are integers (0, 1, 2...)
        # not one-hot encoded vectors.
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model