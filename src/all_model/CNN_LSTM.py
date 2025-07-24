def model_CNN_LSTM(n_lstm, input_shape, num_classes):
    print(f"\n- Xây dựng mô hình CNN-LSTM {n_lstm} LSTM layers")
    
    # Tạo mô hình CNN với n_cnn cố định là 3
    cnn_model = model_CNN(3, input_shape, num_classes)

    # Lấy đầu ra từ tầng Flatten của mô hình CNN
    cnn_output = cnn_model.output

    # Tạo mô hình LSTM
    # Cần reshape đầu ra của CNN để phù hợp với input_shape của LSTM
    lstm_input_shape = (cnn_output.shape[1], cnn_output.shape[2])  # Thay đổi hình dạng đầu vào cho LSTM

    # Tạo mô hình LSTM với tham số n_lstm
    lstm_model = model_LSTM(n_lstm, lstm_input_shape, num_classes)

    # Kết hợp cả hai mô hình
    combined_output = lstm_model(cnn_output)

    # Tạo mô hình cuối cùng
    model = Model(inputs=cnn_model.input, outputs=combined_output)
    
    model.compile(optimizer=RMSprop(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model