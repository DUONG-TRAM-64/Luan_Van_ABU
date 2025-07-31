import pandas as pd

# Danh sách các đặc trưng từ dữ liệu CIC-CSE-IDS-2018
features = ['Dst Port', 'Protocol', 'Timestamp', 'Flow Duration', 'Tot Fwd Pkts',
    'Tot Bwd Pkts', 'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd Pkt Len Max',
    'Fwd Pkt Len Min', 'Fwd Pkt Len Mean', 'Fwd Pkt Len Std',
    'Bwd Pkt Len Max', 'Bwd Pkt Len Min', 'Bwd Pkt Len Mean',
    'Bwd Pkt Len Std', 'Flow Byts/s', 'Flow Pkts/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Tot',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Tot', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max',
    'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags',
    'Bwd URG Flags', 'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s',
    'Bwd Pkts/s', 'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean',
    'Pkt Len Std', 'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
    'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
    'CWE Flag Count', 'ECE Flag Cnt', 'Down/Up Ratio', 'Pkt Size Avg',
    'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
    'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
    'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
    'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
    'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
    'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min', 'Label']

# Phân loại đặc trưng (nhóm tạm thời, có thể mở rộng)
def categorize_feature(feature):
    feature = feature.lower()
    if 'pkt' in feature or 'seg' in feature or 'bytes' in feature:
        return 'Thông tin gói tin'
    elif 'iat' in feature or 'active' in feature or 'idle' in feature:
        return 'Thời gian/độ trễ'
    elif 'flag' in feature or 'urg' in feature or 'psh' in feature:
        return 'Cờ TCP'
    elif 'header' in feature or 'protocol' in feature or 'port' in feature:
        return 'Thông tin kết nối'
    elif 'flow' in feature or 'subflow' in feature:
        return 'Luồng'
    elif 'label' in feature:
        return 'Nhãn'
    elif 'timestamp' in feature:
        return 'Thời gian'
    else:
        return 'Khác'

# Tạo DataFrame mô tả đặc trưng
df = pd.DataFrame({
    'STT': range(1, len(features)+1),
    'Tên đặc trưng': features,
    'Phân loại': [categorize_feature(f) for f in features]
})

# Xuất ra file CSV
output_path = 'Phan_loai_dac_trung_CIC_CSE_IDS2018.csv'
df.to_csv(output_path, index=False)
output_path
