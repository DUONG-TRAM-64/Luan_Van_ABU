import csv
import os

model_list = [ "DNN", "DBN", "MLP"]

for n in range(1, 6):  # Duyệt qua n từ 1 đến 5
    # output_file = f"get_{n}.csv"
    output_file = f"get_class{n}.csv"
    header_written = False

    with open(output_file, 'w', newline='') as fout:
        writer = csv.writer(fout)

        for model_name in model_list:
            filename = f"report_01_{model_name}_{n}_lay.csv"
            
            # filename = f"report_class_{model_name}_{n}_lay.csv"
            if not os.path.exists(filename):
                print(f"Bỏ qua: {filename} không tồn tại.")
                continue

            with open(filename, 'r', newline='') as fin:
                reader = list(csv.reader(fin))
                if not reader:
                    print(f"Bỏ qua: {filename} rỗng.")
                    continue

                if not header_written:
                    writer.writerow(['Model', 'Source_File'] + reader[0])  # Ghi tiêu đề từ dòng đầu tiên của file
                    header_written = True

                writer.writerow([model_name, filename] + reader[-1])  # Ghi dòng cuối