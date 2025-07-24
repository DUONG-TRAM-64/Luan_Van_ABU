# ---------------------------------------------------------------------------- #
#                                   MODEL CNN                                  #
# ---------------------------------------------------------------------------- #
def model_CNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN - {n} Conv1D layers")
    model = Sequential()
    neuron_per_layer = {
        # -------------------------------- 256 neurons ------------------------------- #
        1: [256],
        2: [128, 128],
        3: [64, 64, 128],
        4: [32, 32, 64, 128],
        5: [32, 32, 64, 64, 64]
    }
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    for i, filters in enumerate(layer_list):
        if i == 0:
            model.add(Conv1D(filters, kernel_size=2, activation='relu', padding='same', 
                            input_shape=input_shape, name=f'conv1d_{i+1}'))
        else:
            model.add(Conv1D(filters, kernel_size=2, activation='relu', padding='same', 
                            name=f'conv1d_{i+1}'))
        if n == 5 and i == len(layer_list) - 1:
            model.add(GlobalMaxPooling1D(name=f'global_maxpool_{i+1}'))
        elif i < len(layer_list) - 1:
            model.add(MaxPooling1D(pool_size=2, strides=2, padding='same', name=f'maxpool_{i+1}'))
    
    model.add(BatchNormalization(name='batch_norm'))
    model.add(Dropout(0.5, name='dropout'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                  MODEL LSTM                                  #
# ---------------------------------------------------------------------------- #
def model_LSTM(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình LSTM - {n} LSTM layers")
    model = Sequential()
    neuron_per_layer = {
        1: [512],
        2: [256, 256],
        3: [128, 128, 256],
        4: [64, 64, 128, 256],
        5: [64, 64, 128, 128, 128]
    }
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    for i, units in enumerate(layer_list):
        if i == 0:
            if n == 1:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=False, name=f'lstm_{i+1}'))
            else:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=True, name=f'lstm_{i+1}'))
        elif i == len(layer_list) - 1:
            model.add(LSTM(units, return_sequences=False, name=f'lstm_{i+1}'))
        else:
            model.add(LSTM(units, return_sequences=True, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL GRU                                  #
# ---------------------------------------------------------------------------- #
def model_GRU(n, input_shape, num_classes): 
    print(f"\n\U0001F6E0️ Xây dựng mô hình GRU - {n} GRU layers")
    model = Sequential()
    neuron_per_layer = {
        1: [512],
        2: [256, 256],
        3: [128, 128, 256],
        4: [64, 64, 128, 256],
        5: [64, 64, 128, 128, 128]
    }
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]

    for i, units in enumerate(layer_list):
        if i == 0:
            if n == 1:
                model.add(GRU(units, input_shape=input_shape, return_sequences=False, name=f'gru_{i+1}'))
            else:
                model.add(GRU(units, input_shape=input_shape, return_sequences=True, name=f'gru_{i+1}'))
        elif i == len(layer_list) - 1:
            model.add(GRU(units, return_sequences=False, name=f'gru_{i+1}'))
        else:
            model.add(GRU(units, return_sequences=True, name=f'gru_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL DNN                                  #
# ---------------------------------------------------------------------------- #
def model_DNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình DNN - {n} Dense layers")
    model = Sequential()
    neuron_per_layer = {
        1: [512],
        2: [256, 256],
        3: [128, 128, 256],
        4: [64, 64, 128, 256],
        5: [64, 64, 128, 128, 128]
    }
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")

    layer_list = neuron_per_layer[n]
    for i, units in enumerate(layer_list):
        if i == 0:
            model.add(Dense(units, activation='relu', input_shape=input_shape, name=f'dense_{i+1}'))
        else:
            model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(BatchNormalization(name=f'bn_{i+1}'))
        model.add(Dropout(0.3, name=f'dropout_{i+1}'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(num_classes, activation='softmax', name='output'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                MODEL CNN-LSTM                                #
# ---------------------------------------------------------------------------- #
def model_CNN_LSTM(n_lstm, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN-LSTM {n_lstm} LSTM layers")
    
    cnn_model = model_CNN(3, input_shape, num_classes)
    cnn_output = cnn_model.get_layer('flatten').output
    
    lstm_input_shape = (cnn_output.shape[1], 1)
    cnn_output = Reshape(lstm_input_shape)(cnn_output)
    
    lstm_model = model_LSTM(n_lstm, lstm_input_shape, num_classes)
    combined_output = lstm_model(cnn_output)
    
    model = Model(inputs=cnn_model.input, outputs=combined_output)
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                  MODEL CNN-GRU                               #
# ---------------------------------------------------------------------------- #
def model_CNN_GRU(n_gru, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN-GRU {n_gru} GRU layers")
    
    cnn_model = model_CNN(3, input_shape, num_classes)
    cnn_output = cnn_model.get_layer('flatten').output
    
    gru_input_shape = (cnn_output.shape[1], 1)
    cnn_output = Reshape(gru_input_shape)(cnn_output)
    
    gru_model = model_GRU(n_gru, gru_input_shape, num_classes)
    combined_output = gru_model(cnn_output)
    
    model = Model(inputs=cnn_model.input, outputs=combined_output)
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                  MODEL CNN-DNN                               #
# ---------------------------------------------------------------------------- #
def model_CNN_DNN(n_dnn, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN-DNN {n_dnn} Dense layers")
    
    cnn_model = model_CNN(3, input_shape, num_classes)
    cnn_output = cnn_model.get_layer('flatten').output
    
    dnn_input_shape = (cnn_output.shape[1],)
    dnn_model = model_DNN(n_dnn, dnn_input_shape, num_classes)
    combined_output = dnn_model(cnn_output)
    
    model = Model(inputs=cnn_model.input, outputs=combined_output)
    
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model