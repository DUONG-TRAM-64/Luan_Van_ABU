import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dropout, Flatten, Dense, BatchNormalization, GRU, Input
from keras.optimizers import RMSprop, Adam

# Biến toàn cục cho cấu hình số nơ-ron
neuron_per_layer = {
    1: [256],
    2: [128, 128],
    3: [64, 64, 128],
    4: [32, 32, 64, 128],
    5: [32, 32, 64, 64, 64]
}
#  neuron_per_layer = {
#             1: [512],
#             2: [256, 256],
#             3: [128, 128, 256],
#             4: [64, 64, 128, 256],
#             5: [64, 64, 128, 128, 128]
# }
# ---------------------------------------------------------------------------- #
#                                   MODEL CNN                                  #
# ---------------------------------------------------------------------------- #

def model_CNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN - {n} Conv1D layers")
    model = Sequential()

    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    for i, filters in enumerate(layer_list):
        model.add(Conv1D(filters, kernel_size=2, activation='relu', padding='same', 
                         input_shape=input_shape if i == 0 else None, name=f'conv1d_{i+1}'))
        
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
   
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    for i, units in enumerate(layer_list):
        if i == 0:
            model.add(LSTM(units, input_shape=input_shape, return_sequences=n > 1, name=f'lstm_{i+1}'))
        else:
            model.add(LSTM(units, return_sequences=i < len(layer_list) - 1, name=f'lstm_{i+1}'))
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
    print(f"\n- Xây dựng mô hình GRU - {n} GRU layers")
    model = Sequential()
   
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]

    for i, units in enumerate(layer_list):
        if i == 0:
            model.add(GRU(units, input_shape=input_shape, return_sequences=n > 1, name=f'gru_{i+1}'))
        else:
            model.add(GRU(units, return_sequences=i < len(layer_list) - 1, name=f'gru_{i+1}'))
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

    model.add(Dense(num_classes, activation='softmax', name='output'))
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL DBN                                  #
# ---------------------------------------------------------------------------- #
def model_DBN(n, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình DBN - {n} Dense layers")
    model = Sequential()

    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong neuron_per_layer. Chọn từ 1 đến 5.")

    layer_list = neuron_per_layer[n]

    for i, neurons in enumerate(layer_list):
        if i == 0:
            model.add(Dense(neurons, activation='relu', input_shape=(input_shape[0],), name=f'dense_{i+1}'))
        else:
            model.add(Dense(neurons, activation='relu', name=f'dense_{i+1}'))

        model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(Dropout(0.5, name=f'dropout_{i+1}'))

    model.add(Dense(num_classes, activation='softmax', name='output'))
    model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL MLP                                  #
# ---------------------------------------------------------------------------- #
def model_MLP(n, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình MLP - {n} Dense layers")
    model = Sequential()

    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong neuron_per_layer. Chọn từ 1 đến 5.")

    layer_list = neuron_per_layer[n]
    model.add(Input(shape=(input_shape[0],)))

    for i, neurons in enumerate(layer_list):
        model.add(Dense(neurons, activation='relu', name=f'dense_{i+1}'))
        model.add(BatchNormalization(name=f'batch_norm_{i+1}'))
        model.add(Dropout(0.5, name=f'dropout_{i+1}'))

    model.add(Dense(num_classes, activation='softmax', name='output'))
    model.compile(optimizer=RMSprop(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                MODEL HYBRID                                  #
# ---------------------------------------------------------------------------- #
def model_CNN_LSTM(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-LSTM - {n} LSTM layers")
    model = Sequential()
    
    # Fixed CNN layers
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_1'))
    
    model.add(Conv1D(64, kernel_size=3, activation='relu', name='conv1d_2'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_2'))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', name='conv1d_3'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_3'))
    
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # LSTM layers
    for i, units in enumerate(layer_list):
        model.add(LSTM(units, return_sequences=i < len(layer_list) - 1, name=f'lstm_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    # Final layers
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL CNN-GRU                              #
# ---------------------------------------------------------------------------- #
def model_CNN_GRU(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-GRU - {n} GRU layers")
    model = Sequential()
    
    # Fixed CNN layers
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_1'))
    
    model.add(Conv1D(64, kernel_size=3, activation='relu', name='conv1d_2'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_2'))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', name='conv1d_3'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_3'))
    
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # GRU layers
    for i, units in enumerate(layer_list):
        model.add(GRU(units, return_sequences=i < len(layer_list) - 1, name=f'gru_{i+1}'))
        model.add(Dropout(0.2, name=f'dropout_{i+1}'))
    
    # Final layers
    model.add(BatchNormalization(name='batch_normalization'))
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    # Compile model
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL CNN-DNN                              #
# ---------------------------------------------------------------------------- #
def model_CNN_DNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN-DNN - {n} Dense layers")
    model = Sequential()
    
    # Fixed CNN layers
    model.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape, name='conv1d_1'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_1'))
    
    model.add(Conv1D(64, kernel_size=3, activation='relu', name='conv1d_2'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_2'))
    
    model.add(Conv1D(128, kernel_size=3, activation='relu', name='conv1d_3'))
    model.add(MaxPooling1D(pool_size=2, name='maxpool_3'))
    
    model.add(Flatten(name='flatten'))
    
    if n not in neuron_per_layer:
        raise ValueError(f"Số layer n={n} chưa được định nghĩa trong layer_list. Chọn từ 1 đến 5.")
    layer_list = neuron_per_layer[n]
    
    # Dense layers
    for i, units in enumerate(layer_list):
        model.add(Dense(units, activation='relu', name=f'dense_{i+1}'))
        model.add(Dropout(0.3, name=f'dropout_{i+1}'))
    
    model.add(Dense(num_classes, activation='softmax', name='dense_output'))
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# ---------------------------------------------------------------------------- #
#                                   MODEL NAMES                                 #
# ---------------------------------------------------------------------------- #
def get_model_name(num):
    model_names = {
        1: "CNN", 2: "LSTM", 3: "GRU", 4: "DNN", 5: "CNN-LSTM", 6: "CNN-GRU", 7: "CNN-DNN", 8: "MLP", 9: "DBN"
    }
    return model_names.get(num, "Không tìm thấy mô hình tương ứng")

def build_model(n, model_num, input_shape, num_classes):
    if model_num == 1:
        return model_CNN(n, input_shape, num_classes)
    elif model_num == 2:
        return model_LSTM(n, input_shape, num_classes)
    elif model_num == 3:
        return model_GRU(n, input_shape, num_classes)
    elif model_num == 4:
        return model_DNN(n, input_shape, num_classes)
    elif model_num == 5:
        return model_CNN_LSTM(n, input_shape, num_classes)
    elif model_num == 6:
        return model_CNN_GRU(n, input_shape, num_classes)
    elif model_num == 7:
        return model_CNN_DNN(n, input_shape, num_classes)
    elif model_num == 8:
        return model_DBN(n, input_shape, num_classes)
    elif model_num == 9:
        return model_MLP(n, input_shape, num_classes)
    else:
        raise ValueError("model_num không hợp lệ. Vui lòng chọn từ 1 đến 9.")
