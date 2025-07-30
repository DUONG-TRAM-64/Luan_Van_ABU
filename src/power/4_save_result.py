
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