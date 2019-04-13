
FEA_DIMENSION = 80

label_dict = {'alv':0, 'fas':1, 'prs':2, 'pus':3, 'urd':4, 'mul':5}

path_conf = {
    "dev_data": '',
    "dev_labels": '',
    "train_list":'',
}

log_conf = {
    "base_dir": '',
    "file_name": '',
}

summary_conf = {
    "checkpoint_dir":'',
    "model_directory":'',
    "model_name":'',
}

model_conf = {
    "gru_layer1_units": 256,
    "gru_layer2_units": 256,
    "bigru_units": 256,
    "attention_size":128,
    "dense_dim_3s":512,
    "dense_dim_10s": 512,
    "num_of_classes": 6,
    "learning_rate":1e-4,
}

train_conf = {
    "train_batch_size": 64,
    "val_batch_size": 70,
    "num_classes": 6,
    "num_epochs": 100,
    "num_batches": 1000,
    "window":20,
    "hop":10,
    "sample_len":3010,
}