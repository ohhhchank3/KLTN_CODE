import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import argparse
import logging
import re
import subprocess

LOG_PATH = "./logs/"
LOG_FORMAT = "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

parser = argparse.ArgumentParser()
# ------multi worker-----------------
parser.add_argument('--model-path-address',
                    default="THUDM/chatglm2-6b@localhost@20002",
                    nargs="+",
                    type=str,
                    help="Đường dẫn mô hình, máy chủ và cổng, định dạng là model-path@host@port")
# ---------------controller-------------------------

parser.add_argument("--controller-host", type=str, default="localhost")
parser.add_argument("--controller-port", type=int, default=21001)
parser.add_argument(
    "--dispatch-method",
    type=str,
    choices=["lottery", "shortest_queue"],
    default="shortest_queue",
)
controller_args = ["controller-host", "controller-port", "dispatch-method"]

# ----------------------worker------------------------------------------

parser.add_argument("--worker-host", type=str, default="localhost")
parser.add_argument("--worker-port", type=int, default=21002)
# parser.add_argument("--worker-address", type=str, default="http://localhost:21002")
# parser.add_argument(
#     "--controller-address", type=str, default="http://localhost:21001"
# )
parser.add_argument(
    "--model-path",
    type=str,
    default="lmsys/vicuna-7b-v1.3",
    help="Đường dẫn đến trọng số. Đây có thể là một thư mục cục bộ hoặc một ID repo Hugging Face."
)
parser.add_argument(
    "--revision",
    type=str,
    default="main",
    help="Xác định phiên bản mô hình trên Hugging Face Hub"
)
parser.add_argument(
    "--device",
    type=str,
    choices=["cpu", "cuda", "mps", "xpu"],
    default="cuda",
    help="Loại thiết bị"
)
parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help="Một GPU như 1 hoặc nhiều GPU như 0,2"
)
parser.add_argument("--num-gpus", type=int, default=1)
parser.add_argument(
    "--max-gpu-memory",
    type=str,
    default="20GiB",
    help="Bộ nhớ tối đa trên mỗi GPU. Sử dụng chuỗi như '13Gib'"
)
parser.add_argument(
    "--load-8bit", action="store_true", help="Sử dụng 8-bit quantization"
)
parser.add_argument(
    "--cpu-offloading",
    action="store_true",
    help="Chỉ khi sử dụng 8-bit quantization: Offload trọng số dư ra CPU nếu không vừa GPU"
)
parser.add_argument(
    "--gptq-ckpt",
    type=str,
    default=None,
    help="Load quantized model. Đường dẫn đến checkpoint GPTQ cục bộ."
)
parser.add_argument(
    "--gptq-wbits",
    type=int,
    default=16,
    choices=[2, 3, 4, 8, 16],
    help="Số bit sử dụng cho quantization"
)
parser.add_argument(
    "--gptq-groupsize",
    type=int,
    default=-1,
    help="Kích thước nhóm sử dụng cho quantization; mặc định sử dụng toàn bộ hàng."
)
parser.add_argument(
    "--gptq-act-order",
    action="store_true",
    help="Áp dụng thứ tự activation GPTQ heuristic không"
)
parser.add_argument(
    "--model-names",
    type=lambda s: s.split(","),
    help="Các tên hiển thị tùy chọn, phân tách bằng dấu phẩy"
)
parser.add_argument(
    "--limit-worker-concurrency",
    type=int,
    default=5,
    help="Giới hạn đồng thời của mô hình để tránh OOM."
)
parser.add_argument("--stream-interval", type=int, default=2)
parser.add_argument("--no-register", action="store_true")

worker_args = [
    "worker-host", "worker-port",
    "model-path", "revision", "device", "gpus", "num-gpus",
    "max-gpu-memory", "load-8bit", "cpu-offloading",
    "gptq-ckpt", "gptq-wbits", "gptq-groupsize",
    "gptq-act-order", "model-names", "limit-worker-concurrency",
    "stream-interval", "no-register",
    "controller-address", "worker-address"
]
# -----------------openai server---------------------------

parser.add_argument("--server-host", type=str, default="localhost", help="host name")
parser.add_argument("--server-port", type=int, default=8888, help="port number")
parser.add_argument(
    "--allow-credentials", action="store_true", help="cho phép thông tin xác thực"
)
parser.add_argument(
    "--api-keys",
    type=lambda s: s.split(","),
    help="Danh sách tùy chọn các khóa API, phân tách bằng dấu phẩy",
)
server_args = ["server-host", "server-port", "allow-credentials", "api-keys",
               "controller-address"
               ]

# 0,controller, model_worker, openai_api_server
# 1, các tùy chọn dòng lệnh
# 2, LOG_PATH
# 3, tên file log
base_launch_sh = "nohup python3 -m fastchat.serve.{0} {1} >{2}/{3}.log 2>&1 &"

# 0 log_path
# ! 1 log file name, phải khớp với base_launch_sh
# 2 controller, worker, openai_api_server
base_check_sh = """while [ `grep -c "Uvicorn running on" {0}/{1}.log` -eq '0' ];do
                        sleep 5s;
                        echo "đang chờ {2} chạy"
                done
                echo '{2} đã chạy' """


def string_args(args, args_list):
    """Chuyển các key trong args thành chuỗi"""
    args_str = ""
    for key, value in args._get_kwargs():
        # key trong args._get_kwargs được tách bằng "_", chuyển thành "-" trước khi kiểm tra xem có trong args_list không
        key = key.replace("_", "-")
        if key not in args_list:
            continue
        # Các key port, host trong fastchat không có tiền tố, loại bỏ tiền tố
        key = key.split("-")[-1] if re.search("port|host", key) else key
        if not value:
            pass
        # 1==True ->  True
        elif isinstance(value, bool) and value == True:
            args_str += f" --{key} "
        elif isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            value = " ".join(value)
            args_str += f" --{key} {value} "
        else:
            args_str += f" --{key} {value} "

    return args_str


def launch_worker(item, args, worker_args=worker_args):
    log_name = item.split("/")[-1].split("\\")[-1].replace("-", "_").replace("@", "_").replace(".", "_")
    # Tách model-path-address trước khi chuyển vào string_args để phân tích các tham số
    args.model_path, args.worker_host, args.worker_port = item.split("@")
    args.worker_address = f"http://{args.worker_host}:{args.worker_port}"
    print("*" * 80)
    print(f"Nếu không khởi động lâu, vui lòng kiểm tra logs tại {LOG_PATH}{log_name}.log")
    worker_str_args = string_args(args, worker_args)
    print(worker_str_args)
    worker_sh = base_launch_sh.format("model_worker", worker_str_args, LOG_PATH, f"worker_{log_name}")
    worker_check_sh = base_check_sh.format(LOG_PATH, f"worker_{log_name}", "model_worker")
    subprocess.run(worker_sh, shell=True, check=True)
    subprocess.run(worker_check_sh, shell=True, check=True)


def launch_all(args,
               controller_args=controller_args,
               worker_args=worker_args,
               server_args=server_args
               ):
    print(f"Khởi động dịch vụ llm, logs được lưu tại {LOG_PATH}...")
    controller_str_args = string_args(args, controller_args)
    controller_sh = base_launch_sh.format("controller", controller_str_args, LOG_PATH, "controller")
    controller_check_sh = base_check_sh.format(LOG_PATH, "controller", "controller")
    subprocess.run(controller_sh, shell=True, check=True)
    subprocess.run(controller_check_sh, shell=True, check=True)
    print(f"Thời gian khởi động worker có thể lâu, khoảng 3-10 phút, vui lòng đợi...")
    if isinstance(args.model_path_address, str):
        launch_worker(args.model_path_address, args=args, worker_args=worker_args)
    else:
        for idx, item in enumerate(args.model_path_address):
            print(f"Khởi động mô hình thứ {idx}:{item}")
            launch_worker(item, args=args, worker_args=worker_args)

    server_str_args = string_args(args, server_args)
    server_sh = base_launch_sh.format("openai_api_server", server_str_args, LOG_PATH, "openai_api_server")
    server_check_sh = base_check_sh.format(LOG_PATH, "openai_api_server", "openai_api_server")
    subprocess.run(server_sh, shell=True, check=True)
    subprocess.run(server_check_sh, shell=True, check=True)
    print("Khởi động dịch vụ llm hoàn tất!")


if __name__ == "__main__":
    args = parser.parse_args()
    args = argparse.Namespace(**vars(args),
                              **{"controller-address": f"http://{args.controller_host}:{str(args.controller_port)}"})

    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Lớn hơn --num-gpus ({args.num_gpus}) so với --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    launch_all(args=args)
