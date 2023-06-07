## 项目结构

```
.
├── data
│   ├── annotations
│   └── zip_feats
├── inference.sh
├── init.sh
├── src
│   ├── category_id_map.py
│   ├── config.py
│   ├── data_helper.py
│   ├── inference.py
│   ├── job1
│   │   ├── README.md
│   │   ├── category_id_map.py
│   │   ├── config.py
│   │   ├── data_helper.py
│   │   ├── evaluate.py
│   │   ├── log.txt
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── requirements.txt
│   │   ├── swa_from_ema.py
│   │   └── util.py
│   ├── job2
│   │   ├── README.md
│   │   ├── __pycache__
│   │   │   ├── category_id_map.cpython-39.pyc
│   │   │   ├── config.cpython-39.pyc
│   │   │   ├── data_helper.cpython-39.pyc
│   │   │   ├── model.cpython-39.pyc
│   │   │   └── util.cpython-39.pyc
│   │   ├── category_id_map.py
│   │   ├── config.py
│   │   ├── data_helper.py
│   │   ├── evaluate.py
│   │   ├── log.txt
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── requirements.txt
│   │   └── util.py
│   ├── model.py
│   ├── pre_train
│   │   ├── requirements.txt
│   │   └── train_0614_1431.log
│   └── third_party
│       └── qq_unibert
│           ├── model
│           │   ├── __pycache__
│           │   │   └── model_uni.cpython-39.pyc
│           │   └── model_uni.py
│           └── pre_train
│               ├── category_id_map.py
│               ├── config.json
│               ├── config.py
│               ├── create_optimizer.py
│               ├── data_cfg.py
│               ├── masklm.py
│               ├── model_cfg.py
│               ├── pretrain.py
│               ├── pretrain_cfg.py
│               ├── qq_dataset.py
│               ├── qq_uni_model.py
│               ├── requirements.txt
│               ├── train_0614_1431.log
│               └── utils.py
└── train.sh
```

