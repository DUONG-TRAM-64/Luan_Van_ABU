def model_DNN(n, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình DNN - {n} Dense layers")
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