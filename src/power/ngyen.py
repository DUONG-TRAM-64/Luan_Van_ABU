import pandas as pd
import numpy as np
import time
import os
import shutil
import gc
import subprocess
import psutil
import GPUtil
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, Dropout, Flatten, Dense, BatchNormalization, GRU, Input
from keras.optimizers import RMSprop, Adam

# ---------------------------------------------------------------------------- #
#                                DATA PROCESSING                               #
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
    
    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test
# ---------------------------------------------------------------------------- #
#                                 COnfig model                                 #
# ---------------------------------------------------------------------------- #
def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("\n- Bắt đầu quá trình huấn luyện...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=125, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-7, verbose=1)
    ]
    start = time.time()
    history = model.fit(X_train, y_train, batch_size=512, epochs=50,
                        validation_data=(X_val, y_val), callbacks=callbacks,
                        class_weight=class_weights, verbose=1)
    train_time = time.time() - start
    print(f"- Thời gian huấn luyện: {train_time:.2f} giây")
    return history, train_time

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
    
# ---------------------------------------------------------------------------- #
#                            RESOURCE MONITORING                               #
# ---------------------------------------------------------------------------- #
def get_gpu_memory_used_nvidia_smi():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader'],
            capture_output=True, text=True, check=True
        )
        memory_str = result.stdout.strip()
        if memory_str:
            return float(memory_str.split(' ')[0])
        return 0.0
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Cảnh báo: Không thể lấy thông tin bộ nhớ GPU qua nvidia-smi: {e}")
        return 0.0
    except Exception as e:
        print(f"Cảnh báo: Lỗi không xác định khi lấy bộ nhớ GPU: {e}")
        return 0.0

def start_monitor(output_dir, model_name, n_layers):
    gpu_log_file = os.path.join(output_dir, f"gpu_usage_log_{model_name}_{n_layers}_layers.csv")
    print(f"- File log GPU cho model này sẽ được lưu tại: {gpu_log_file}")
    if os.path.exists(gpu_log_file):
        os.remove(gpu_log_file)
        print(f"- Đã xóa file log GPU cũ: {gpu_log_file}")
    nvidia_smi_command = [
        "nvidia-smi",
        "--loop=1",
        f"--query-gpu=timestamp,power.draw,temperature.gpu,utilization.gpu,utilization.memory",
        "--format=csv,noheader"
    ]
    process = None
    gpu_log_fd = None
    try:
        gpu_log_fd = open(gpu_log_file, 'a')
        process = subprocess.Popen(
            nvidia_smi_command,
            stdout=gpu_log_fd,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )
        print("- Bắt đầu ghi log sử dụng GPU...")
    except FileNotFoundError:
        print("Lỗi: 'nvidia-smi' không tìm thấy. Đảm bảo nó được cài đặt và trong PATH.")
    except Exception as e:
        print(f"Lỗi khi bắt đầu ghi log GPU: {e}")
    p = psutil.Process(os.getpid())
    ram_before = p.memory_info().rss / (1024 * 1024) 
    gpus = GPUtil.getGPUs()
    gpu_mem_before = gpus[0].memoryUsed if gpus else get_gpu_memory_used_nvidia_smi()
    start_time = time.time()
    return process, ram_before, gpu_mem_before, start_time, gpu_log_fd

def end_monitor(process, ram_before, gpu_mem_before, start_time, gpu_log_fd):
    total_time = time.time() - start_time
    print(f"- Dừng ghi log GPU và tính toán tài nguyên sử dụng ({total_time:.2f}s)...")
    if process:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Cảnh báo: Tiến trình nvidia-smi không dừng trong 5 giây, đang kill.")
            process.kill()
        if gpu_log_fd:
            gpu_log_fd.close()
    else:
        print("- Tiến trình ghi log GPU chưa bao giờ được khởi tạo hoặc đã gặp lỗi.")
    p = psutil.Process(os.getpid())
    ram_after = p.memory_info().rss / (1024 * 1024) 
    ram_increase = ram_after - ram_before
    gpus = GPUtil.getGPUs()
    gpu_mem_after = gpus[0].memoryUsed if gpus else get_gpu_memory_used_nvidia_smi()
    gpu_ram_increase = gpu_mem_after - gpu_mem_before
    print(f"  - RAM hệ thống tăng: {ram_increase:.2f} MB")
    if gpus:
        print(f"  - GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB")
    else:
        print("  - Không tìm thấy GPU hoặc không sử dụng GPU.")
    return total_time, ram_increase, gpu_ram_increase

# ---------------------------------------------------------------------------- #
#                                Evalute on TEST                               #
# ---------------------------------------------------------------------------- #
def evaluate_model(model, X_test, y_test_cat, y_true, label_names, output_dir, num_layers):
    print("\n- Đánh giá mô hình...")   
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    infer_time = time.time() - start
    y_pred_classes = np.argmax(y_pred, axis=1)
   
    print(f"- Accuracy: {test_accuracy * 100:.2f}%")
    print(f"- Suy luận trung bình: {infer_time / X_test.shape[0] * 1000:.4f} ms")
   
    # Generate classification report
    print("\n- Classification Report:")
    report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred_classes, digits=4))
    
    # Convert report to DataFrame and save to CSV
    report_df = pd.DataFrame(report).transpose()
    report_file_path = f"{output_dir}/classification_report_layers_{num_layers}.csv"
    report_df.to_csv(report_file_path, index=True)
    print(f"- Classification report saved to {report_file_path}")
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    return test_accuracy, infer_time, y_pred_classes, report, cm

# ---------------------------------------------------------------------------- #
#                               Training_history                               #
# ---------------------------------------------------------------------------- #
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
# ---------------------------------------------------------------------------- #
#                              confusion_matrices                              #
# ---------------------------------------------------------------------------- #
def save_confusion_matrices(cm, label_names, output_dir, n, model_num):
    model_name = get_model_name(model_num)
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Counts ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    count_path = os.path.join(output_dir, f'{model_name}_{n}_layers_matrix.png')
    plt.savefig(count_path)
    plt.close()
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(np.round(cm_percent, 2), annot=True, fmt='.2f', cmap='YlGnBu',
                      xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Percentage ({model_name} - {n} Layers)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    percent_path = os.path.join(output_dir, f'{model_name}_{n}_layers_matrix_percent.png')
    plt.savefig(percent_path)
    plt.close()
    print(f"Saved confusion matrices for {model_name} with {n} layers.")

# ---------------------------------------------------------------------------- #
#                                 Power report                                 #
# ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
#                                    REPORT                                    #
# ---------------------------------------------------------------------------- #

def save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase):
    model_name = get_model_name(model_num)
    gpu_log_file = os.path.join(output_dir, f"gpu_usage_log_{model_name}_{n}_layers.csv")
    # Khởi tạo giá trị mặc định cho các thông số GPU
    avg_gpu_util = 0.0
    max_gpu_mem_util = 0.0
    avg_gpu_temp = 0.0
    avg_power_draw = 0.0
    
    # Đọc và xử lý file log GPU nếu tồn tại
    if os.path.exists(gpu_log_file):
        try:
            gpu_data = pd.read_csv(gpu_log_file, names=['timestamp', 'power_draw', 'temperature_gpu', 'utilization_gpu', 'utilization_memory'])
            # Làm sạch và chuyển đổi dữ liệu
            gpu_data['power_draw'] = gpu_data['power_draw'].str.replace(' W', '').astype(float)
            gpu_data['temperature_gpu'] = gpu_data['temperature_gpu'].astype(float)
            gpu_data['utilization_gpu'] = gpu_data['utilization_gpu'].str.replace(' %', '').astype(float)
            gpu_data['utilization_memory'] = gpu_data['utilization_memory'].str.replace(' %', '').astype(float) 
            # Tính toán các chỉ số
            avg_gpu_util = gpu_data['utilization_gpu'].mean()
            max_gpu_mem_util = gpu_data['utilization_memory'].max()
            avg_gpu_temp = gpu_data['temperature_gpu'].mean()
            avg_power_draw = gpu_data['power_draw'].mean()
        except Exception as e:
            print(f"Cảnh báo: Không thể đọc hoặc xử lý file log GPU {gpu_log_file}: {e}")

    # Lưu báo cáo dưới dạng file văn bản
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
        f.write(f"- RAM tăng: {ram_increase:.2f} MB\n")
        f.write(f"- GPU RAM đã dùng thêm: {gpu_ram_increase:.2f} MB\n")
        f.write(f"- Mức sử dụng GPU trung bình: {avg_gpu_util:.2f} %\n")
        f.write(f"- Mức sử dụng bộ nhớ GPU tối đa: {max_gpu_mem_util:.2f} %\n")
        f.write(f"- Nhiệt độ GPU trung bình: {avg_gpu_temp:.2f} °C\n")
        f.write(f"- Công suất tiêu thụ trung bình: {avg_power_draw:.2f} W\n")
    
    print(f"Đã lưu báo cáo văn bản cho {model_name} với {n} layers vào file report_{model_name}_{n}_layers.txt.")

    # Lưu báo cáo dưới dạng file CSV
    report_data = {
        "Model_Name": [model_name],
        "Num_Layers": [n],
        "Best_Val_Accuracy": [max(history.history['val_accuracy'])],
        "Test_Accuracy": [test_accuracy],
        "Train_Time_s": [train_time],
        "Inference_Time_s": [infer_time],
        "Avg_Inference_ms": [infer_time / X_test.shape[0] * 1000],
        "Total_Time_s": [total_time],
        "RAM_Increase_MB": [ram_increase],
        "GPU_RAM_Increase_MB": [gpu_ram_increase],
        "Avg_GPU_Util_%": [avg_gpu_util],
        "Max_GPU_Mem_Util_%": [max_gpu_mem_util],
        "Avg_GPU_Temp_C": [avg_gpu_temp],
        "Avg_Power_Draw_W": [avg_power_draw]
    }
    report_df = pd.DataFrame(report_data)
    csv_file_path = f"{output_dir}/report_{model_name}_{n}_layers.csv"
    report_df.to_csv(csv_file_path, index=False)
    print(f"Đã lưu báo cáo CSV cho {model_name} với {n} layers vào file {csv_file_path}.")   

def plot_gpu_usage(file_path, model_name, num_layers, output_dir):
    if not os.path.exists(file_path):
        print(f"Cảnh báo: File log GPU {file_path} không tồn tại. Bỏ qua việc vẽ biểu đồ.")
        return
    try:
        df = pd.read_csv(file_path, names=['Timestamp', 'Power_Draw_W', 'Temperature_C', 'GPU_Utilization_percent', 'Memory_Utilization_percent'])
        df = df.drop(columns=['Timestamp'])
        df['Step'] = range(len(df))
    except Exception as e:
        print(f"Lỗi khi đọc file log GPU {file_path}: {e}")
        return
    plt.rcParams['font.family'] = 'DejaVu Sans'
    metrics = [
        ('GPU_Utilization_percent', '% Sử dụng GPU', 'green', 'Mức độ sử dụng GPU'),
        ('Memory_Utilization_percent', '% Sử dụng bộ nhớ GPU', 'blue', 'Mức độ sử dụng Memory'),
        ('Temperature_C', 'Nhiệt độ GPU (°C)', 'red', 'Nhiệt độ GPU'),
        ('Power_Draw_W', 'Công suất tiêu thụ (W)', 'purple', 'Công suất GPU')
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    axes = axes.flatten()
    for i, (column, ylabel, color, title_suffix) in enumerate(metrics):
        ax = axes[i]
        ax.plot(df['Step'], df[column], color=color, linewidth=1.5)
        ax.set_title(f'{title_suffix} ({model_name} - {num_layers} layers)')
        ax.set_xlabel('Lần ghi (Step)')
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        if "percent" in column.lower():
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'gpu_usage_plot_{model_name}_{num_layers}_layers.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu biểu đồ vào {output_file}")


def main():
    print("- Bắt đầu quá trình huấn luyện các mô hình...")
    output_dir = "/kaggle/working/output"
    if os.path.exists(output_dir):
        print(f"- Đang xóa thư mục output cũ: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"- Thư mục output mới đã được tạo: {output_dir}")
    train_df, test_df = load_data()  # Tải dữ liệu một lần
    model_nums = [1]  # CNN, DNN, CNN-LSTM, CNN-GRU, CNN-DNN, DBN, MLP
    layers_list = [1, 2, 3, 4, 5]
    
    for model_num in model_nums:
        model_name = get_model_name(model_num)
        for n in layers_list:-
            print(f"\n=== Bắt đầu huấn luyện mô hình {model_name} với {n} layer ===")
            # Bắt đầu theo dõi tài nguyên
            process, ram_before, gpu_mem_before, start_time, gpu_log_fd = start_monitor(output_dir, model_name, n)
            # Xử lý dữ liệu
            X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test = preprocess_data(train_df, test_df)
            input_shape = X_train.shape[1:]
            num_classes = y_train_cat.shape[1]
            # Xây dựng và huấn luyện mô hình
            model = build_model(n, model_num, input_shape, num_classes)
            history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
            # Đánh giá mô hình
            test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n)
            # Kết thúc theo dõi tài nguyên
            total_time, ram_increase, gpu_ram_increase = end_monitor(process, ram_before, gpu_mem_before, start_time, gpu_log_fd)
            # Lưu kết quả
            save_training_history(history, output_dir, n, model_num)
            save_confusion_matrices(cm, label_names, output_dir, n, model_num)
            save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase)

            # Vẽ và lưu biểu đồ sử dụng GPU
            plot_gpu_usage(
                file_path=os.path.join(output_dir, f"gpu_usage_log_{model_name}_{n}_layers.csv"),
                model_name=model_name,
                num_layers=n,
                output_dir=output_dir
            )
            
            print(f"\n=== Kết thúc mô hình {model_name} với {n} layer ===")
            del model, history, y_pred_classes, report, cm, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test
            gc.collect()
            print(f"- Đã xóa các biến tạm thời sau khi train mô hình {model_name} {n} layer(s)")
    
    gc.collect()
    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()