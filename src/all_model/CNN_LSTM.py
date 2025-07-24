

def model_CNN_LSTM(n, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN-LSTM - 3 Conv1D layers [32, 64, 128] và {n} LSTM layers")

    model = Sequential()

    # --- 3 Conv1D layers ---
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape, name='conv1d_1'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_1'))

    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', name='conv1d_2'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_2'))

    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same', name='conv1d_3'))
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name='maxpool_3'))

    # --- LSTM Layers ---
    # Tùy chọn số lớp và số units
    neuron_per_layer_lstm = {
        1: [256],
        2: [128, 256],
        3: [128, 128, 256],
        4: [64, 64, 128, 256],
        5: [64, 64, 128, 128, 256]
    }

    if n not in neuron_per_layer_lstm:
        raise ValueError(f"Số lớp LSTM {n} không hợp lệ. Hãy chọn từ 1 đến 5.")

    lstm_layers = neuron_per_layer_lstm[n]

    # Truyền trực tiếp output của Conv1D vào LSTM (không GlobalPooling)
    for i, units in enumerate(lstm_layers):
        return_seq = (i < len(lstm_layers) - 1)
        if i == 0:
            model.add(LSTM(units, return_sequences=return_seq, name=f'lstm_{i+1}'))
        else:
            model.add(LSTM(units, return_sequences=return_seq, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))

    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))

    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    return model
