import os
from sacred import Experiment

ex = Experiment("baseline", save_git_info=False)
 
@ex.config
def my_config():
    track = "main"
    split = '1'
    dataset = "ntu60"
    lr = 0.05 # 0.05 for ntu60 and pku, 0.005 for ntu120
    margin = 0.1
    weight_decay = 0.0005
    epoch_num = 50
    batch_size = 128 #128
    loss_type = "kl"
    alpha = 1
    beta = 1
    m = 1
    DA = True # DA means using our prototype-guided text feature alignment
    fix_encoder = False
    finetune = False
    support_factor = 0.9 # 0.9 for ntu60, 0.4 for ntu120, 1.0 for pku
    weight_path = './module/gcn/model/split_'+split+".pt" # only using when set fix_encoder/finetune
    log_path = './output/log/split_{}_{}_DA_des_support_factor{}_lr{}.log'.format(split,loss_type,support_factor,lr)
    # log_path = './output/log/sota_split10_des_DA_epoch100_lr{}_support_factor{}.log'.format(lr, support_factor)

    save_path = './output/model/split_{}_{}_DA_des_support_factor{}_lr{}.pt'.format(split,loss_type,support_factor,lr)
    loss_mode = "step" # "step" or "cos"
    step = [50, 80]
    ############################## ST-GCN ###############################
    in_channels = 3
    hidden_channels = 16
    hidden_dim = 256
    dropout = 0.5
    graph_args = {
    "layout" : 'ntu-rgb+d',
    "strategy" : 'spatial'
    }
    edge_importance_weighting = True
    ############################# downstream #############################
    split_1 = [4,19,31,47,51]
    split_2 = [12,29,32,44,59]
    split_3 = [7,20,28,39,58]
    split_4 = [3, 18, 26, 38, 41, 60, 87, 99, 102, 110]
    split_5 = [5, 12, 14, 15, 17, 42, 67, 82, 100, 119]
    split_6 = [6, 20, 27, 33, 42, 55, 71, 97, 104, 118]
    split_7 = [1, 9, 20, 34, 50]
    split_8 = [3, 14, 29, 31, 49]
    split_9 = [2, 15, 39, 41, 43]
    unseen_label = eval('split_'+split)
    visual_size = 256
    language_size = 768
    max_frame = 50
    language_path = "./data/language/"+dataset+"_des_embeddings.npy" # des best
    train_list = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_data.npy"
    train_label = "./data/zeroshot/"+dataset+"/split_"+split+"/seen_train_label.npy"
    test_list = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_data.npy"
    test_label = "./data/zeroshot/"+dataset+"/split_"+split+"/unseen_label.npy"
    ############################ sota compare ############################
    sota_split = "5" # 5 or 12 or 10 or 24 
    model_choice_for_sota = 'shift-gcn' # shift-gcn or st-gcn
    unseen_label_5 = [10,11,19,26,56]
    unseen_label_12 = [3,5,9,12,15,40,42,47,51,56,58,59]
    unseen_label_10 = [4,13,37,43,49,65,88,95,99,106]
    unseen_label_24 = [5,9,11,16,18,20,22,29,35,39,45,49,59,68,70,81,84,87,93,94,104,113,114,119]
    sota_unseen = eval('unseen_label_'+sota_split)
    sota_train_list = "./data/zeroshot/"+dataset+"/unseen_label_"+sota_split+"/seen_train_data.npy"
    sota_train_label = "./data/zeroshot/"+dataset+"/unseen_label_"+sota_split+"/seen_train_label.npy"
    sota_test_list = "./data/zeroshot/"+dataset+"/unseen_label_"+sota_split+"/unseen_data.npy"
    sota_test_label = "./data/zeroshot/"+dataset+"/unseen_label_"+sota_split+"/unseen_label.npy"
# %%
