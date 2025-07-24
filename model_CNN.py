def model_CNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN - {n} Conv1D layers")
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
    # ------------------------------ check 1,2,3,4,5 ----------------------------- #
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