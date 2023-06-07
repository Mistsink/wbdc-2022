#! /bin/bash

cd src
# 安装环境 ：因为改推理阶段的环境是 job1 的环境，与 job2 不同
pip install -r requirements.txt
python inference.py