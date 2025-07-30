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