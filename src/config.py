class Arg:
    n_splits = 5
    shuffle = True
    seed = 2022
    dropout = 0.2

    version = 11

    
    
    path_in = '../data/'
    path_out = f'../data/'
    bert_dir = '../data/chinese-roberta-wwm-ext'

#     path_in = './data/'
#     path_out = f'./out/{version}/'
#     bert_dir = './pre_model/chinese-roberta-wwm-ext'

    
    log_path = path_out + 'log.txt'
    log_save = True
    pre_train_path = './pre_train/3_model_pretrain_1.pth'
# ==========================  Crucial  =============================
    batch_size = 32
    val_ratio = 0.1
    max_epochs = 5
    learning_rate = 1e-4
    linear_learning_rate = 5e-4

    text_parts = ['title', 'asr', 'ocr']
    bert_seq_length = 512

    
    warmup_ratio = 0.1


    best_score = 0.6
  
    ckpt_file = path_out + '0.bin'

# ========================== Swa Layer =============================
    use_fgm = True
    fgm_epsilon = 0.5
    
# ========================== Ema Layer =============================
    use_ema = True
    ema_decay = 0.999

# ========================== Swa Layer =============================
    use_swa = True
    swa_start = 1


# ========================= Data Configs ==========================
    train_annotation = path_in + 'annotations/labeled.json'
    test_annotation = path_in + 'annotations/test_b.json'
    train_zip_feats = path_in + 'zip_feats/labeled.zip'
    test_zip_feats = path_in + 'zip_feats/test_b.zip'
    test_output_csv = path_out + 'result_vote.csv'
    
    
    reined_ratio = 1
    
    
    val_batch_size = 256
    test_batch_size = 256
    prefetch = 16
    num_workers = 4
    
    tfidf_voca_path = path_in + 'vocabulary.pkl'
    tfidf_transformer_path = path_in + 'transformer.pkl'
    tfidf_model_path = path_in + 'model.pkl'
    tfidf_voca = ''
    tfidf_model = ''
    tfidf_size = 687690
    tfidf_out_size = 512
    use_tfidf = False    
    tfidf_path = f'{path_in}idf.txt'
    tfidf_top_k = 5


# ======================== SavedModel Configs =========================
    savedmodel_path = path_out

# ========================= Learning Configs ==========================
    
    max_steps = -1
    
    print_steps = 20
    minimum_lr = 0.
    
    weight_decay = 0.01
    adam_epsilon = 1e-6


# ========================== Tokenize Text =============================
    # 90%长度 +asr : 281    + asr + ocr: 573
    
    with_tags = False
    wash_text = False


# ========================== Title BERT =============================
#         bert_dir = './pre_model/macbert-base'
    #bert_dir = './roberta_base'

    bert_cache = 'data/cache'
    bert_learning_rate = 3e-5
    bert_warmup_steps = 1000
    bert_max_steps = 30000
    bert_hidden_dropout_prob = 0.1


# ========================== Video =============================
    frame_embedding_size = 768
    max_frames = 32
    vlad_cluster_size = 64
    vlad_groups = 8
    vlad_hidden_size = 1024
    se_ratio = 8


# ========================== Fusion Layer =============================
    fc_size = 512

def parse_args():

    return Arg()
