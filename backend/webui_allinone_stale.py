"""Usage

python webui_allinone.py

api
python webui_allinone.py --use-remote-api

webui
python webui_allinone.py --nohup


python webui_allinone.py --model-path-address model1@host1@port1 model2@host2@port2 


python webui_alline.py --model-path-address model@host@port --num-gpus 2 --gpus 0,1 --max-gpu-memory 4GiB

"""
import os
import subprocess

import streamlit as st
from streamlit_option_menu import option_menu
from webui_pages.utils import *

from backend.api_allinone_stale import api_args, parser
from backend.llm_api_stale import (LOG_PATH, controller_args, launch_all,
                                   server_args, string_args, worker_args)
from webui_pages import *

parser.add_argument("--use-remote-api", action="store_true")
parser.add_argument("--nohup", action="store_true")
parser.add_argument("--server.port", type=int, default=8501)
parser.add_argument("--theme.base", type=str, default='"light"')
parser.add_argument("--theme.primaryColor", type=str, default='"#165dff"')
parser.add_argument("--theme.secondaryBackgroundColor", type=str, default='"#f5f5f5"')
parser.add_argument("--theme.textColor", type=str, default='"#000000"')
web_args = ["server.port", "theme.base", "theme.primaryColor", "theme.secondaryBackgroundColor", "theme.textColor"]


def launch_api(args, args_list=api_args, log_name=None):
    print("Launching api ...")
    print("Đang khởi chạy dịch vụ API...")
    if not log_name:
        log_name = f"{LOG_PATH}api_{args.api_host}_{args.api_port}"
    print(f"logs on api are written in {log_name}")
    print(f"API logs are written in {log_name}, please check the logs if there are any startup issues")
    args_str = string_args(args, args_list)
    api_sh = "python server/{script} {args_str} >{log_name}.log 2>&1 &".format(
        script="api.py", args_str=args_str, log_name=log_name)
    subprocess.run(api_sh, shell=True, check=True)
    print("launch api done!")
    print("Dịch vụ API đã được khởi chạy thành công.")


def launch_webui(args, args_list=web_args, log_name=None):
    print("Launching webui...")
    print("Đang khởi chạy dịch vụ webui...")
    if not log_name:
        log_name = f"{LOG_PATH}webui"

    args_str = string_args(args, args_list)
    if args.nohup:
        print(f"logs on api are written in {log_name}")
        print(f"webui service logs are written in {log_name}, please check the logs if there are any startup issues")
        webui_sh = "streamlit run webui.py {args_str} >{log_name}.log 2>&1 &".format(
            args_str=args_str, log_name=log_name)
    else:
        webui_sh = "streamlit run webui.py {args_str}".format(
            args_str=args_str)
    subprocess.run(webui_sh, shell=True, check=True)
    print("launch webui done!")
    print("Dịch vụ webui đã được khởi chạy thành công.")


if __name__ == "__main__":
    print("Starting webui_allineone.py, it would take a while, please be patient....")
    print(f"Khởi động webui_allinone, việc khởi chạy dịch vụ LLM mất khoảng từ 3-10 phút, vui lòng kiên nhẫn chờ đợi, nếu lâu quá không khởi động được, vui lòng kiểm tra logs tại {LOG_PATH}...")
    args = parser.parse_args()

    print("*" * 80)
    if not args.use_remote_api:
        launch_all(args=args, controller_args=controller_args, worker_args=worker_args, server_args=server_args)
    launch_api(args=args, args_list=api_args)
    launch_webui(args=args, args_list=web_args)
    print("Start webui_allinone.py done!")
    print("Khởi động webui_allinone hoàn tất. Cảm ơn đã đợi.") 
