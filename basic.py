
# ---------------------------------- func-01 ---------------------------------- #
def load_data():
    print("\n-Đang đọc dữ liệu...")
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
    # --------------------------- Xem Phan bo cac class -------------------------- #
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
    # y : class 1,2,3,4,5,6...
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    
    # Reshape cho CNN hoặc LSTM --> Type : 3 DIM dạng (samples, timesteps, features)
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
        # --->Tao file report ghi lai ket qua 
 # ---------------------------------------------------------------------------- #
 #   You can PASTE các FUNC in .py nay               save_resul.py              #
 # ---------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------- #
#                            func - buil model                                 #
# ---------------------------------------------------------------------------- #
# Hàm gọi hàm phu hop with model_num 
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
# --> Gọi hàm Built model da xay dung 

  # ---------------------------------------------------------------------------- #
  #               You can PASTE các FUNC in .py nay    all_model.py              #
  # ---------------------------------------------------------------------------- #



# ---------------------------------------------------------------------------- #
#                                  func - main                                 #
# ---------------------------------------------------------------------------- #
def main():
    print("- Bắt đầu quá trình huấn luyện nhiều mô hình...")
    output_dir = "/kaggle/working/output"
    os.makedirs(output_dir, exist_ok=True)
    # ---------------------------------- func01 ---------------------------------- #
    train_df, test_df = load_data()
    # ---------------------------------- func02 ---------------------------------- #
    X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test, test_df = preprocess_data(train_df, test_df)
    
    input_shape = X_train.shape[1:]
    num_classes = y_train_cat.shape[1]
    # Chọn mô hình
    model_num = 1  # 1: CNN, 2: LSTM, 3: GRU, 4: DNN, 5: CNN-LSTM, 6: CNN-GRU, 7: GRU-CNN
    layers_list = [1, 2, 3, 4, 5]
    
    for n in layers_list:
        print(f"\n=== Bắt đầu huấn luyện mô hình với {n} layer ===")
        # Gọi hàm xây dựng mô hình
        model = build_model(n, model_num, input_shape, num_classes)
        
        # ---------------------------------- func03-training ---------------------------------- #
        history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
        
        # ------------------------------ func04 -evaluate ----------------------------- #
        test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n, test_df)
        
        # ---------------------------- Saving cac Results ---------------------------- #
        save_training_history(history, output_dir, n, model_num)
        save_confusion_matrices(cm, label_names, output_dir, n, model_num)
        save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num)
        
        # Delete df --> save Ram 
        del model, history, y_pred_classes, report, cm
        gc.collect()
        print(f"- Đã xóa các biến tạm thời sau khi train 1 time : model {n} layer(s)")
        
        print(f"============ Kết thúc mô hình với {n} layer ============")
    
    del X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test, train_df, test_df
    gc.collect()

    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()
