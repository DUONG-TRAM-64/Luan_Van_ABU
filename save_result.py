
def get_model_name(num):
    model_names = {
        1: "CNN", 
        2: "LSTM", 5: "CNN-LSTM",
        3: "GRU",  6: "CNN-GRU",
        4: "DNN",  7: "CNN-DNN"
    }

    return model_names.get(num, "Không tìm thấy mô hình tương ứng")

def save_training_history(history, output_dir, n, model_num):
    model_name = get_model_name(model_num)  # Lấy tên mô hình từ model_num
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title(f"Accuracy ({model_name} - {n} Layers)")  # Sử dụng tên mô hình và số lớp
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title(f"Loss ({model_name} - {n} Layers)")  # Sử dụng tên mô hình và số lớp
    plt.grid(True)

    plt.savefig(f"{output_dir}/training_history_{model_name}_{n}_layers.png")
    plt.close()
    print(f"Đã lưu ảnh training history cho {model_name} với {n} layers.")



def save_confusion_matrices(cm, label_names, output_dir, n, model_num):
    model_name = get_model_name(model_num)  # Lấy tên mô hình từ model_num    
    # Ma trận nhầm lẫn - Số lượng
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                 xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Counts ({model_name} - {n} Layers)')  # Sử dụng tên mô hình và số lớp
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_counts_{model_name}_{n}_layers.png")
    plt.close()
    # Ma trận nhầm lẫn - Tỷ lệ phần trăm
    plt.figure(figsize=(10, 8))
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    sns.heatmap(np.round(cm_percent, 2), annot=True, fmt='.2f', cmap='YlGnBu',
                 xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - Percentage ({model_name} - {n} Layers)')  # Sử dụng tên mô hình và số lớp
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confusion_matrix_percent_{model_name}_{n}_layers.png")
    plt.close()
    print(f"Đã lưu ảnh confusion matrix cho {model_name} với {n} layers.")


def save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num):
    model_name = get_model_name(model_num)  
    report_filename = f"{output_dir}/report_{model_name}_{n}_layers.txt"  # Đặt tên tệp với tên mô hình

    with open(report_filename, 'w') as f:
        f.write("TÓM TẮT KẾT QUẢ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Tên mô hình: {model_name}\n")  # Thêm tên mô hình vào báo cáo
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
        f.write("\nClassification Report:\n")
        f.write(json.dumps(report, indent=4))  # Ghi báo cáo phân loại
        f.write("\n" + "=" * 50)

    print(f"Đã lưu report cho {model_name} với {n} layers vào file {report_filename}.")



save_training_history(history, output_dir, n , model_num)
save_confusion_matrices(cm, label_names, output_dir, n , model_num)
save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir , model_num)