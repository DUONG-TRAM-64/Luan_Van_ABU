def model_LSTM(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình LSTM - {n} LSTM layers")
    model = Sequential()
    neuron_per_layer = {
        # -------------------------------- 256 neurons ------------------------------- #
        # 1: [256],
        # 2: [128, 128],
        # 3: [64, 64, 128],
        # 4: [32, 32, 64, 128],
        # 5: [32, 32, 64, 64, 64]
        # -------------------------------- 512 neurons ------------------------------- #
        1: [512],
        2: [256, 256],
        3: [128, 128, 256],
        4: [64, 64, 128, 256],
        5: [64, 64, 128, 128, 128]
    }
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # Thêm các tầng LSTM
    for i, units in enumerate(layer_list):
        if i == 0:
            # Tầng đầu tiên cần input_shape
            if n == 1:
                # Nếu chỉ có 1 tầng, không trả về sequences
                model.add(LSTM(units, input_shape=input_shape, return_sequences=False, name=f'lstm_{i+1}'))
            else:
                model.add(LSTM(units, input_shape=input_shape, return_sequences=True, name=f'lstm_{i+1}'))
        elif i == len(layer_list) - 1:
            # Tầng cuối không trả về sequences
            model.add(LSTM(units, return_sequences=False, name=f'lstm_{i+1}'))
        else:
            # Các tầng giữa trả về sequences
            model.add(LSTM(units, return_sequences=True, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

