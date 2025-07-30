# ---------------------------------------------------------------------------- #
#                                DATA PROCESSING                               #
# ---------------------------------------------------------------------------- #
def load_data():
    print("\n- Đang đọc dữ liệu...")
    
    # train_df = pd.read_parquet('/kaggle/input/150-iot23/train_other.parquet')
    # test_df = pd.read_parquet('/kaggle/input/150-iot23/test_other.parquet')
    train_df = pd.read_parquet('/kaggle/input/150-cse-new/train_cse_150.parquet')
    test_df = pd.read_parquet('/kaggle/input/150-cse-new/test_cse.parquet')
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    train_df = train_df.sample(frac=0.01, random_state=511).reset_index(drop=True) 
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



