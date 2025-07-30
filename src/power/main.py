# ---------------------------------------------------------------------------- #
#                                 MAIN FUNCTION                                #
# ---------------------------------------------------------------------------- #
def main():
    print("- Bắt đầu quá trình huấn luyện các mô hình...")
    output_dir = "/kaggle/working/output"
    if os.path.exists(output_dir):
        print(f"- Đang xóa thư mục output cũ: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"- Thư mục output mới đã được tạo: {output_dir}")
    model_nums = [1,4,5,6,7,8,9]
    #model_nums = [2,3]
    layers_list = [1, 2, 3, 4, 5,]
    for model_num in model_nums:
        model_name = get_model_name(model_num)
        for n in layers_list:
            print(f"\n=== Bắt đầu huấn luyện mô hình {model_name} với {n} layer ===")
            # Start Prcess
            process, ram_before, gpu_mem_before, start_time, gpu_log_fd = start_monitor(output_dir, model_name, n)
            # ---------------------------------------------------------------- Train model -------------------------------------------------------------- #
            train_df, test_df = load_data()
            X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, class_weights, label_names, y_test = preprocess_data(train_df, test_df)
            input_shape = X_train.shape[1:]
            num_classes = y_train_cat.shape[1]
            model = build_model(n, model_num, input_shape, num_classes)
            history, train_time = train_model(model, X_train, y_train_cat, X_val, y_val_cat, class_weights)
            test_accuracy, infer_time, y_pred_classes, report, cm = evaluate_model(model, X_test, y_test_cat, y_test, label_names, output_dir, n)
            # ENDs Prcess
            total_time, ram_increase, gpu_ram_increase = end_monitor(process, ram_before, gpu_mem_before, start_time, gpu_log_fd)
            # ----------------------------------------------------------- save all result ---------------------------------------------------------- #
            save_training_history(history, output_dir, n, model_num)
            save_confusion_matrices(cm, label_names, output_dir, n, model_num)
            save_report(n, test_accuracy, infer_time, train_time, history, model, X_train, X_test, report, output_dir, model_num, total_time, ram_increase, gpu_ram_increase)
            
            
            print(f"\n=== Kết thúc mô hình {model_name} với {n} layer ===")
            del model, history, y_pred_classes, report, cm, X_train, y_train_cat, X_val, y_val_cat, X_test, y_test_cat, y_test, train_df, test_df
            gc.collect()
            print(f"- Đã xóa các biến tạm thời sau khi train mô hình {model_name} {n} layer(s)")
    gc.collect()
    print("- Đã xóa dữ liệu sau khi hoàn thành tất cả model.")

if __name__ == '__main__':
    main()
