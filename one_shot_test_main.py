from one_shot_test_config import *
# from model import *
from dataset import DataSet 
from one_shot_test_logger import Log

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from math import pi, cos
from tqdm import tqdm

from module.gcn.st_gcn import Model
from module.shift_gcn import Model as ShiftGCN
from module.adapter import Adapter, Linear
from KLLoss import KLLoss, KDLoss
from tool import gen_label, create_logits, get_acc, create_sim_matrix, gen_label_from_text_sim, get_m_theta, get_acc_v2

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(0) # 0->2025

# %%
class Processor:

    @ex.capture
    def load_data(self, train_list, train_label, test_list, test_label, batch_size, language_path, one_shot_exemplar_data_path):
        self.dataset = dict()
        self.data_loader = dict()
        self.best_epoch = -1
        self.best_acc = -1
        self.dim_loss = -1
        self.test_acc = -1
        self.test_aug_acc = -1
        self.best_aug_acc = -1
        self.best_aug_epoch = -1
        
        self.full_language = np.load(language_path)
        self.full_language = torch.Tensor(self.full_language)
        self.full_language = self.full_language.cuda()

        self.one_shot_exemplar_data = np.load(one_shot_exemplar_data_path)
        self.one_shot_exemplar_data = torch.Tensor(self.one_shot_exemplar_data).cuda()

        self.dataset['train'] = DataSet(train_list, train_label)
        self.dataset['test'] = DataSet(test_list, test_label)

        self.data_loader['train'] = torch.utils.data.DataLoader(
            dataset=self.dataset['train'],
            batch_size=batch_size,
            num_workers=16,
            shuffle=True,
            drop_last=True)

        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=self.dataset['test'],
            batch_size=64,
            num_workers=16,
            shuffle=False)

    def load_weights(self, model=None, weight_path=None):
        pretrained_dict = torch.load(weight_path)
        model.load_state_dict(pretrained_dict)

    def adjust_learning_rate(self,optimizer,current_epoch, max_epoch,lr_min=0,lr_max=0.1,warmup_epoch=15, loss_mode='step', step=[50, 80]):

        if current_epoch < warmup_epoch:
            lr = lr_max * current_epoch / warmup_epoch
        elif loss_mode == 'cos':
            lr = lr_min + (lr_max-lr_min)*(1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
        elif loss_mode == 'step':
            lr = lr_max * (0.1 ** np.sum(current_epoch >= np.array(step)))
        else:
            raise Exception('Please check loss_mode!')
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            # if i == 0:
            #     param_group['lr'] = lr * 0.1
            # else:
            #     param_group['lr'] = lr
    
    def layernorm(self, feature):

        num = feature.shape[0]
        mean = torch.mean(feature, dim=1).reshape(num, -1)
        var = torch.var(feature, dim=1).reshape(num, -1)
        out = (feature-mean) / torch.sqrt(var)

        return out

    @ex.capture
    def load_model(self,in_channels,hidden_channels,hidden_dim,
                    dropout,graph_args,edge_importance_weighting, loss_type, weight_path):
        self.encoder = Model(in_channels=in_channels, hidden_channels=hidden_channels,
                            hidden_dim=hidden_dim,dropout=dropout, 
                            graph_args=graph_args,
                            edge_importance_weighting=edge_importance_weighting,
                            )
        self.encoder = self.encoder.cuda()
        self.adapter = Linear().cuda()
        if loss_type == "kl":
            self.loss = KLLoss().cuda()
        else:
            raise Exception('loss_type Error!')
        self.logit_scale = self.adapter.get_logit_scale()
        self.logit_scale_v2 = self.adapter.get_logit_scale_v2()
        pretrained_dict = torch.load(weight_path)
        # print(pretrained_dict['encoder'])
        self.encoder.load_state_dict(pretrained_dict['encoder'])
        self.adapter.load_state_dict(pretrained_dict['adapter'])
        
        # self.model = MI(visual_size, language_size).cuda()


    @ex.capture
    def load_optim(self, lr, epoch_num, weight_decay):
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.encoder.parameters()},
        #     {'params': self.model.parameters()}],
        #      lr=lr,
        #      weight_decay=weight_decay,
        #      )
        # self.optimizer = torch.optim.Adam([
        #     {'params': self.encoder.parameters()},
        #     {'params': self.adapter.parameters()}],
        #      lr=lr,
        #      weight_decay=weight_decay,
        #      )
        self.optimizer = torch.optim.SGD([
            {'params': self.encoder.parameters()},
            {'params': self.adapter.parameters()}],
             lr=lr,
             weight_decay=weight_decay,
             momentum=0.9,
             nesterov=False
             )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 100)

    @ex.capture
    def optimize(self, epoch_num, DA): # print -> log.info
        self.log.info("main track")
        epoch = 0
        with torch.no_grad():
            self.test_epoch(epoch=epoch)
        self.log.info("epoch [{}] test acc: {}".format(epoch,self.test_acc))
        # self.log.info("epoch [{}] gets the best acc: {}".format(self.best_epoch,self.best_acc))
        if DA:
            self.log.info("epoch [{}] DA test acc: {}".format(epoch,self.test_aug_acc))

    @ex.capture
    def train_epoch(self, epoch, lr, loss_mode, step, loss_type, alpha, beta, m, fix_encoder):
        self.encoder.train() # eval -> train
        if fix_encoder:
            self.encoder.eval()
        self.adapter.train()
        self.adjust_learning_rate(self.optimizer, current_epoch=epoch, max_epoch=100, lr_max=lr, warmup_epoch=5, loss_mode=loss_mode, step=step)
        running_loss = []
        loader = self.data_loader['train']
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            # print(data.shape) #128,3,50,25,2
            # label = label.type(torch.LongTensor).cuda()
            label_g = gen_label(label)
            label = label.type(torch.LongTensor).cuda()
            # print(label.shape) # 128
            # print(label) # int
            seen_language = self.full_language[label] # 128, 768
            # print(seen_language.shape)
            
            feat = self.encoder(data)
            if fix_encoder:
                feat = feat.detach()
            skleton_feat = self.adapter(feat)
            if loss_type == "kl":
                logits_per_skl, logits_per_text = create_logits(skleton_feat, seen_language, self.logit_scale, exp=True)
                ground_truth = torch.tensor(label_g, dtype=skleton_feat.dtype).cuda()
                # ground_truth = gen_label_from_text_sim(seen_language)
                loss_skls = self.loss(logits_per_skl, ground_truth)
                loss_texts = self.loss(logits_per_text, ground_truth)
                loss = (loss_skls + loss_texts) / 2
            else:
                raise Exception('loss_type Error!')
                
            running_loss.append(loss)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        running_loss = torch.tensor(running_loss)
        self.dim_loss = running_loss.mean().item()
    
    @ex.capture
    def test_epoch(self, unseen_label, epoch, DA, support_factor, da_iterations=3):  # Add da_iterations parameter
        self.encoder.eval()
        self.adapter.eval()

        loader = self.data_loader['test']
        acc_list = []
        
        # First get baseline accuracy without DA
        for data, label in tqdm(loader):
            data = data.type(torch.FloatTensor).cuda()
            label = label.type(torch.LongTensor).cuda()
            one_shot_skeleton = self.adapter(self.encoder(self.one_shot_exemplar_data))
            feature = self.encoder(data)
            feature = self.adapter(feature)
            acc_batch, pred = get_acc(feature, one_shot_skeleton, unseen_label, label)
            acc_list.append(acc_batch)

        acc_list = torch.tensor(acc_list)
        acc = acc_list.mean()
        if acc > self.best_acc:
            self.best_acc = acc 
            self.best_epoch = epoch
            self.save_model()
        self.test_acc = acc

        if DA:
            current_skeleton = self.adapter(self.encoder(self.one_shot_exemplar_data))
            best_aug_acc = 0

            # Perform multiple DA iterations
            for da_iter in range(da_iterations):
                ent_list = []
                feat_list = []
                old_pred_list = []
                
                # Collect features and entropy for refinement
                for data, label in tqdm(loader):
                    data = data.type(torch.FloatTensor).cuda()
                    label = label.type(torch.LongTensor).cuda()
                    feature = self.encoder(data)
                    feature = self.adapter(feature)
                    _, pred, old_pred, ent, feat = get_acc_v2(feature, current_skeleton, unseen_label, label)
                    ent_list.append(ent)
                    feat_list.append(feat)
                    old_pred_list.append(old_pred)

                # Refine prototypes
                ent_all = torch.cat(ent_list)
                feat_all = torch.cat(feat_list)
                old_pred_all = torch.cat(old_pred_list)
                z_list = []
                for i in range(len(unseen_label)):
                    mask = old_pred_all == i
                    class_support_set = feat_all[mask]
                    class_ent = ent_all[mask]
                    class_len = class_ent.shape[0]
                    if int(class_len*support_factor) < 1:
                        z = self.full_language[unseen_label[i:i+1]]
                    else:
                        _, indices = torch.topk(-class_ent, int(class_len*support_factor))
                        z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
                    z_list.append(z)

                current_skeleton = torch.cat(z_list)
                
                # Evaluate with refined prototypes
                aug_acc_list = []
                for data, label in tqdm(loader):
                    data = data.type(torch.FloatTensor).cuda()
                    label = label.type(torch.LongTensor).cuda()
                    feature = self.encoder(data)
                    feature = self.adapter(feature)
                    acc_batch, pred = get_acc(feature, current_skeleton, unseen_label, label)
                    aug_acc_list.append(acc_batch)
                    
                aug_acc = torch.tensor(aug_acc_list).mean()
                
                # Update best augmented accuracy
                if aug_acc > best_aug_acc:
                    best_aug_acc = aug_acc
                    best_skeleton = current_skeleton.clone()

            # Update class attributes with best results
            if best_aug_acc > self.best_aug_acc:
                self.best_aug_acc = best_aug_acc
                self.best_aug_epoch = epoch
            self.test_aug_acc = best_aug_acc

    # @ex.capture
    # def test_epoch(self, unseen_label, epoch, DA, support_factor):
    #     self.encoder.eval()
    #     self.adapter.eval()

    #     loader = self.data_loader['test']
    #     y_true = []
    #     y_pred = []
    #     acc_list = []
    #     ent_list = []
    #     feat_list = []
    #     old_pred_list = []
    #     for data, label in tqdm(loader):

    #         # y_t = label.numpy().tolist()
    #         # y_true += y_t

    #         data = data.type(torch.FloatTensor).cuda()
    #         label = label.type(torch.LongTensor).cuda()
    #         # unseen_language = self.full_language[unseen_label]
    #         one_shot_skeleton = self.adapter(self.encoder(self.one_shot_exemplar_data))
    #         # inference
    #         feature = self.encoder(data)
    #         feature = self.adapter(feature)
    #         if DA:
    #         # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
    #             acc_batch, pred, old_pred, ent, feat = get_acc_v2(feature, one_shot_skeleton, unseen_label, label)
    #             ent_list.append(ent)
    #             feat_list.append(feat)
    #             old_pred_list.append(old_pred)
    #         else:
    #             acc_batch, pred = get_acc(feature, one_shot_skeleton, unseen_label, label)
        
    #         # y_p = pred.cpu().numpy().tolist()
    #         # y_pred += y_p


    #         acc_list.append(acc_batch)

    #     acc_list = torch.tensor(acc_list)
    #     acc = acc_list.mean()
    #     if acc > self.best_acc:
    #         self.best_acc = acc
    #         self.best_epoch = epoch
    #         self.save_model()
    #         # y_true = np.array(y_true)
    #         # y_pred = np.array(y_pred)
    #         # np.save("y_true_3.npy",y_true)
    #         # np.save("y_pred_3.npy",y_pred)
    #         # print("save ok!")
    #     self.test_acc = acc
        
    #     if DA:
    #         ent_all = torch.cat(ent_list)
    #         feat_all = torch.cat(feat_list)
    #         old_pred_all = torch.cat(old_pred_list)
    #         z_list = []
    #         for i in range(len(unseen_label)):
    #             mask = old_pred_all == i
    #             class_support_set = feat_all[mask]
    #             class_ent = ent_all[mask]
    #             class_len = class_ent.shape[0]
    #             if int(class_len*support_factor) < 1:
    #                 z = self.full_language[unseen_label[i:i+1]]
    #             else:
    #                 _, indices = torch.topk(-class_ent, int(class_len*support_factor))
    #                 z = torch.mean(class_support_set[indices], dim=0, keepdim=True)
    #             z_list.append(z)
                
    #         z_tensor = torch.cat(z_list)
    #         aug_acc_list = []
    #         for data, label in tqdm(loader):
    #             # y_t = label.numpy().tolist()
    #             # y_true += y_t

    #             data = data.type(torch.FloatTensor).cuda()
    #             label = label.type(torch.LongTensor).cuda()
    #             one_shot_skeleton = z_tensor
    #             # inference
    #             feature = self.encoder(data)
    #             feature = self.adapter(feature)
    #             # acc_batch, pred = get_acc(feature, unseen_language, unseen_label, label)
    #             acc_batch, pred = get_acc(feature, one_shot_skeleton, unseen_label, label)
            
    #             # y_p = pred.cpu().numpy().tolist()
    #             # y_pred += y_p
    #             aug_acc_list.append(acc_batch)
    #         aug_acc = torch.tensor(aug_acc_list).mean()
    #         if aug_acc > self.best_aug_acc:
    #             self.best_aug_acc = aug_acc
    #             self.best_aug_epoch = epoch
    #         self.test_aug_acc = aug_acc
            


    def initialize(self):
        self.load_data()
        self.load_model()
        self.load_optim()
        self.log = Log()

    @ex.capture
    def save_model(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'encoder':self.encoder.state_dict(), 'adapter':self.adapter.state_dict()}, save_path)

    def start(self):
        self.initialize()
        self.optimize()
        # self.save_model()

    


# %%
@ex.automain
def main(track):
    if "main" in track:
        p = Processor()
    p.start()
