import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import json
import gc
import psutil
import GPUtil
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, GRU, Reshape
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------------------------------------------------------------------------- #
#                                   MODEL CNN                                  #
# ---------------------------------------------------------------------------- #
def model_CNN(n, input_shape, num_classes): 
    print(f"\n- Xây dựng mô hình CNN - {n} Conv1D layers")
    model = Sequential()
    neuron_per_layer = {
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

# ---------------------------------------------------------------------------- #
#                               DATA PROCESSING                                #
# ---------------------------------------------------------------------------- #
def load_data():
    print("\n- Đang đọc dữ liệu...")
    train_df = pd.read_parquet('/kaggle/input/150-cse-new/train_cse_150.parquet')
    test_df = pd.read_parquet('/kaggle/input/150-cse-new/test_cse.parquet')
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    train_df = train_df.sample(frac=1, random_state=511).reset_index(drop=True)
    return train_df, test_df

def preprocess_data(train_df, test_df):
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=112)
    print("Validation shape:", val_df.shape)
    print("\n- Phân bố nhãn trong tập train:")
    print(train_df['Label'].value_counts())
    print("\n- Phân bố nhãn trong tập validation:")
    print(val_df['Label'].value_counts())
    print("\n- Phân bố nhãn trong tập test:")
    print(test_df['Label'].value_counts())
    
    X_train = train_df.drop(['Label', 'Label_encode'], axis=1).values
    y_train = train_df['Label_encode'].values
    X_val = val_df.drop(['Label', 'Label_encode'], axis=1).values
    y_val = val_df['Label_encode'].values
    X_test = test_df.drop(['Label', 'Label_encode'], axis=1).values
    y_test = test_df['Label_encode'].values
    
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    classes = np.array(sorted(train_df['Label_encode'].unique()))
    class_weights_arr = class_weight.compute_class_weight('balanced', classes=classes, y=train_df['Label_encode'].values)
    class_weights = dict(zip(classes, class_weights_arr))
    label_names = list(train_df.sort_values('Label_encode')['Label'].unique())
    
    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test, test_df

# ---------------------------------------------------------------------------- #
#                               RESOURCE MONITORING                            #
# ---------------------------------------------------------------------------- #
def start_monitor():
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss
    gpus = GPUtil.getGPUs()
    gpu_mem_before = gpus[0].memoryUsed if gpus else 0
    start_time = time.time()
    return process, ram_before, gpu_mem_before, start_time

def end_monitor(process, ram_before, gpu_mem_before, start_time):
    ram_after = process.memory_info().rss
    gpus = GPUtil.getGPUs()
    gpu_mem_after = gpus[0].memoryUsed if gpus else 0
    end_time = time.time()
    
    print("\n- Tổng kết tài nguyên:")
    total_time = end_time - start_time
    ram_increase = ram_after - ram_before
    gpu_ram_increase = gpu_mem_after - gpu_mem_before
    print(f"- Tổng thời gian chạy: {total_time:.2f} giây")
    print(f"- RAM tăng: {ram_increase / (1024 ** 2):.2f} MB")
    if gpus:
        print(f"- GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB")
    else:
        print("- Không tìm thấy GPU hoặc không sử dụng.")
    return total_time, ram_increase, gpu_ram_increase

# ---------------------------------------------------------------------------- #
#                               MODEL UTILITIES                                #
# ---------------------------------------------------------------------------- #
def get_model_name(num):
    model_names = {
        1: "CNN", 2: "LSTM", 3: "GRU", 4: "DNN",
        5: "CNN-LSTM", 6: "CNN-GRU", 7: "CNN-DNN"
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
    else:
        raise ValueError("model_num không hợp lệ. Vui lòng chọn từ 1 đến 7.")

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("\n- Bắt đầu quá trình huấn luyện...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    start = time.time()
    history = model.fit(X_train, y_train, batch_size=512, epochs=100,
                       validation_data=(X_val, y_val), callbacks=callbacks,
                       class_weight=class_weights, verbose=1)
    train_time = time.time() - start
    print(f"- Thời gian huấn luyện: {train_time:.2f} giây")
    return history, train_time

def evaluate_model(model, X_test, y_test_cat, y_true, label_names, output_dir, num_layers):
    print("\n- Đánh giá mô hình...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    infer_time = time.time() - start
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(f"- Accuracy: {test_accuracy*100:.2f}%")
    print(f"- Suy luận trung bình: {infer_time / X_test.shape[0] * 1000:.4f} ms")
    print("\n- Classification Report:")
    report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred_classes, digits=4))
    cm = confusion_matrix(y_true, y_pred_classes)
    return test_accuracy, infer_time, y_pred_classes, report, cm

def save_training_history(history, output_dir, n, model_num):
    model_name = get_model_name(model_num)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title(f"Accuracy ({model_name} - {n} Layers)")
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f"Loss ({model_name} - {n} Layers)")
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_history_{model_name}_{n}_layers.png")
    plt.close()
    print(f"Đã lưu ảnh training history cho {model_name} với {n} layers.")

def save_confusion_matrices(cm, label_names, output_dir, n, model_num):
    model_name = get_model_name(model_num)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Counts ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_counts_{model_name}_{n}_layers.png")
    plt.close()
    
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(np.round(cm_percent, 2), annot=True, fmt='.2f', cmap='YlGnBu', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Percentage ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_percent_{model_name}_{n}_layers.png")
    plt.close()
    print(f"Đã lưu ảnh confusion matrix cho {model_name} với {n} layers.")

def save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase):
    model_name = get_model_name(model_num)
    with open(f"{output_dir}/report_{model_name}_{n}_layers.txt", 'w') as f:
        f.write(f"TÓM TẮT KẾT QUẢ - {model_name}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Tên mô hình: {model_name}\n")
        f.write(f"Số layer: {n}\n")
        f.write(f"Số mẫu train: {X_train.shape[0]:,}\n")
        f.write(f"Số mẫu test: {X_test.shape[0]:,}\n")
        f.write(f"Số epoch đã train: {len(history.history['loss'])}\n")
        f.write(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}\n")
        f.write(f"Test accuracy cuối cùng: {test_accuracy:.4f}\n")
        f.write(f"Số lượng tham số: {model.count_params():,}\n")
        f.write(f"Thời gian huấn luyện: {train_time:.2f} giây\n")
        f.write(f"Tổng thời gian suy luận: {infer_time:.2f} giây\n")
        f.write(f"Suy luận trung bình mỗi mẫu: {infer_time / X_test.shape[0] * 1000:.4f} ms\n")
        f.write("\n- Tổng kết tài nguyên:\n")
        f.write(f"- Tổng thời gian chạy: {total_time:.2f} giây\n")
        f.write(f"- RAM tăng: {ram_increase / (1024 ** 2):.2f} MB\n")
        f.write(f"- GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB\n")
        f.write("\nClassification Report:\n")
        f.write(json.dumps(report, indent=4))
        f.write("\n" + "=" * 50)
    print(f"Đã lưu report cho {model_name} với {n} layers vào file report_{model_name}_{n}_layers.txt.")

# ---------------------------------------------------------------------------- #
#                                    MAIN                                      #
# ---------------------------------------------------------------------------- #
def main():
    print("- Bắt đầu quá trình huấn luyện các mô hình...")
    output_dir = "/kaggle/working/output"
    os.makedirs(output_dir, exist_ok=True)
    
    train_df, test_df = load_data()
    X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test, test_df = preprocess_data(train_df, test_df)
    
    input_shape = X_train.shape[1:]
    num_classes = y_train_cat.shape[1]
    layers_list = [1, 2, 3, 4, 5]
    
    for model_num in range(1, 8):  # Loop through all 7 models
        for n in layers_list:
            print(f"\n=== Bắt đầu huấn luyện mô hình {get_model_name(model_num)} với {n} layer ===")
            process, ram_before, gpu_mem_before, start_time = start_monitor()
            
            model = build_model(n, model_num, input_shape, num_classes)
            history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
            test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n)
            total_time, ram_increase, gpu_ram_increase = end_monitor(process, ram_before, gpu_mem_before, start_time)
            
            save_training_history(history, output_dir, n, model_num)
            save_confusion_matrices(cm, label_names, output_dir, n, model_num)
            save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase)
            
            print(f"\n=== Kết thúc mô hình {get_model_name(model_num)} với {n} layer ===")
            del model, history, y_pred_classes, report, cm
            gc.collect()
            print(f"- Đã xóa các biến tạm thời sau khi train mô hình {get_model_name(model_num)} {n} layer(s)")
    
    del X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test, train_df, test_df
    gc.collect()
    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()