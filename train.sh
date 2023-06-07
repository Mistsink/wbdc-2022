#! /bin/bash



# =======  预训练阶段
cd src/pre_train
# 安装环境
pip install -r requirements.txt

# 预训练 -- 使用 QQ浏览器比赛中开源的 unibert
cd ../third_party/qq_unibert/pre_train
python pretrain.py



# =======  训练第一个模型
cd ../../../job1
# 安装环境      由于该模型与预训练环境相同，故不再重复安装
# pip install -r requirements.txt

# 训练模型      采用了 ema + swa + fgm
python main.py
python swa_from_ema.py      # 根据生成的 ema 模型进行 swa



# =======  训练第二个模型
cd ../job2
# 安装环境
pip install -r requirements.txt

# 训练模型      采用了 swa + fgm
python main.py


