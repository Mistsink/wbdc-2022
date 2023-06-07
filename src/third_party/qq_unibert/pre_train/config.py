class Arg:
    seed = 2022
    dropout = 0.3

    version = 4

    path_in = '../../../../data/'
    path_out = f'../../../../data/'
    log_path = path_out + 'log.txt'
    log_save = True

# ==========================  Crucial  =============================
    batch_size = 32
    val_ratio = 0.01
    max_epochs = 5
    learning_rate = 7e-5


    text_parts = ['title', 'asr', 'ocr']
    bert_seq_length = 512 - 32


    warmup_ratio = 0.1


    best_score = 0.6
  
    ckpt_file = path_out + 'epoch_5_f1_0.6813.bin'

# ========================== Ema Layer =============================
    use_ema = False
    ema_decay = 0.995

# ========================== Swa Layer =============================
    use_swa = True
    swa_start = 2


# ========================= Data Configs ==========================
    train_annotation = path_in + 'annotations/labeled.json'
    test_annotation = path_in + 'annotations/test_a.json'
    unlabeled_annotation = path_in + 'annotations/unlabeled.json'
    train_zip_feats = path_in + 'zip_feats/labeled.zip'
    test_zip_feats = path_in + 'zip_feats/test_a.zip'
    unlabeled_zip_feats = path_in + 'zip_feats/unlabeled.zip'
    test_output_csv = path_out + 'result.csv'
    
    
    reined_ratio = 1
    
    
    val_batch_size = 64
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
    
    weight_decay = 0.1
    adam_epsilon = 1e-6


# ========================== Tokenize Text =============================
    # 90%长度 +asr : 281    + asr + ocr: 573
    
    with_tags = False
    wash_text = False


# ========================== Title BERT =============================
#         bert_dir = './pre_model/macbert-base'
    bert_dir = '../../../../data/chinese-roberta-wwm-ext'

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
