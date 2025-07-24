
# ---------------------------------- func-01 ---------------------------------- #
def load_data():
    print("\U0001F4E5 Đang đọc dữ liệu...")
    
    train_df = pd.read_parquet('/kaggle/input/150-cse-new/train_cse_150.parquet')
    test_df = pd.read_parquet('/kaggle/input/150-cse-new/test_cse.parquet')
    
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    train_df = train_df.sample(frac=1, random_state=511).reset_index(drop=True)
    return train_df, test_df
# --------------------------------- func - 02 -------------------------------- #
def preprocess_data(train_df, test_df):
    train_df, val_df = train_test_split(train_df, test_size=0.125, random_state=112)

    print("Validation shape:", val_df.shape)

    print("\n Phân bố nhãn trong tập train:")
    print(train_df['Label'].value_counts())
    
    print("\n Phân bố nhãn trong tập validation:")
    print(val_df['Label'].value_counts())
    
    print("\n Phân bố nhãn trong tập test:")
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
    
    # Reshape cho CNN hoặc LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    # Tính class Weight
    classes = np.array(sorted(train_df['Label_encode'].unique()))
    class_weights_arr = class_weight.compute_class_weight('balanced', 
                                                          classes=classes,
                                                          y=train_df['Label_encode'].values)
    class_weights = dict(zip(classes, class_weights_arr))
    
    label_names = list(train_df.sort_values('Label_encode')['Label'].unique())
    
    return X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test, test_df
# --------------------------------- func - 03 -------------------------------- #
def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    print("\n Bắt đầu quá trình huấn luyện...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)
    ]
    start = time.time()
    history = model.fit(X_train, y_train, 
                        batch_size=512,
                        epochs=100,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks,
                        class_weight=class_weights,
                        verbose=1)
    end = time.time()
    print(f"-Thời gian huấn luyện: {end - start:.2f} giây")
    return history, end - start
# --------------------------------- func - 04 -------------------------------- #
def evaluate_model(model, X_test, y_test_cat, y_true, label_names, output_dir, num_layers, test_df):
    print("\n- Đánh giá mô hình...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    start = time.time()
    y_pred = model.predict(X_test, verbose=0)
    end = time.time()

    y_pred_classes = np.argmax(y_pred, axis=1)
    infer_time = end - start
    print(f"- Accuracy: {test_accuracy*100:.2f}%")
    print(f"- Suy luận trung bình: {infer_time / X_test.shape[0] * 1000:.4f} ms")
    print("\n- Classification Report:")
    # Tao Report
    report = classification_report(y_true, y_pred_classes, digits=4, output_dict=True)
    print(classification_report(y_true, y_pred_classes, digits=4))
    # Tao Ma tran nham lan dang so 
    cm = confusion_matrix(y_true, y_pred_classes)
    return test_accuracy, infer_time, y_pred_classes, report, cm

# --------------------------------- func - 05 -------------------------------- #
# Tao file report ghi lai ket qua 
def save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir):
    with open(f"{output_dir}/report_{n}_layers.txt", 'w') as f:
        f.write("TÓM TẮT KẾT QUẢ\n")
        f.write("=" * 50 + "\n")
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
        f.write(json.dumps(report, indent=4))
        f.write("\n" + "=" * 50)
    # ghi thêm vào tính Power
    print(f"Đã lưu report cho {n} layers vào file report_{n}_layers.txt.")


# ---------------------------------------------------------------------------- #
#                            func - buil model - CNN                           #
# ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                                  func - main                                 #
# ---------------------------------------------------------------------------- #
def main():
    print("Bắt đầu quá trình huấn luyện nhiều mô hình...")
    #path OUTPUT
    output_dir = "/kaggle/working/output"
    os.makedirs(output_dir, exist_ok=True)
    # ---------------------------------- func01 ---------------------------------- #
    train_df, test_df = load_data()
    # ---------------------------------- func02 ---------------------------------- #
    X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test, test_df = preprocess_data(train_df, test_df)
    # Dòng  - classes
    input_shape = X_train.shape[1:]
    num_classes = y_train_cat.shape[1]
    # Training 1 times : 1 ,2  3, ,4 ,5 layer
    layers_list = [1, 2, 3, 4, 5]
    
    for n in layers_list:
        print(f"\n=== Bắt đầu huấn luyện mô hình với {n} layer ===")
        
        # --------------- Call func BUILT model : CNN  ,LSTM, DNN, .... -------------- #
        model = build_model_12345_layer(n, input_shape, num_classes)  
        # ---------------------------------- func03-training ---------------------------------- #
        history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
        # ------------------------------ func04 -evalute ----------------------------- #
        test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n, test_df)
        # ---------------------------- Saving cac Results ---------------------------- #
        save_training_history(history, output_dir, n)
        save_confusion_matrices(cm, label_names, output_dir, n)
        save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir)
        # Delete df --> save Ram 
        del model, history, y_pred_classes, report, cm
        gc.collect()
        print(f"-Đã xóa các biến tạm thời sau khi train 1 time : model {n} layer(s)")
        
        print(f"============ Kết thúc mô hình với {n} layer ============")
    
    del X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test, train_df, test_df
    gc.collect()

    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()
