import numpy
import torch
import math

mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
def gen_label(labels):
    num = len(labels)
    gt = numpy.zeros(shape=(num,num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i,k] = 1
    return gt

def gen_label_from_text_sim(x):
    x = x / x.norm(dim=-1, keepdim=True)
    return x @ x.t()

def get_m_theta(cos_theta, m=4):
    cos_m_theta = mlambda[m](cos_theta)
    temp = cos_theta.clone().detach()
    theta = torch.acos(temp.clamp(-1.+1e-6, 1.-1e-6))
    k = (theta*m / math.pi).floor()
    sign = -2 * torch.remainder(k, 2) + 1  # (-1)**k
    phi_theta = sign * cos_m_theta - 2. * k
    return phi_theta
    # d_theta = phi_theta - cos_theta
    # return d_theta + x


def create_logits(x1, x2, logit_scale, exp=True):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    if exp:
        scale = logit_scale.exp()
    else:
        scale = logit_scale

    # cosine similarity as logits
    logits_per_x1 = scale * x1 @ x2.t()
    logits_per_x2 = logits_per_x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2

def create_sim_matrix(x1, x2, alpha=1):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    x1x1 = alpha * x1 @ x1.t()
    x1x2 = alpha * x1 @ x2.t()
    x2x2 = alpha * x2 @ x2.t()
    return x1x1,x1x2,x2x2
    

def get_acc(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred

def get_acc_v2(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    unseen_len = len(unseen_label)
    
    old_pred = pred
    ent = softmax_entropy(logits)
    
    # unseen_len = len(unseen_label)
    # for i in range(unseen_len):
    #     class_support_set = x1[pred == i]
    #     class_logit = logits[pred == i]
    #     class_ent = softmax_entropy(class_logit)
    #     _, indices = torch.topk(class_ent, 5)
    #     z = torch.mean(class_support_set[indices], dim=-1)
    #     z_list.append(z)
    
        
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, old_pred, ent, x1

def get_acc_v3(x1, x2, unseen_label, label):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)
    logits = x1 @ x2.t() # 128, 5
    pred = torch.argmax(logits, dim=1)
    ent = softmax_entropy(logits)
    unseen_label = torch.tensor(unseen_label).cuda()
    pred = torch.index_select(unseen_label,0,pred)
    acc = pred.eq(label.view_as(pred)).float().mean()
    return acc, pred, ent

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

# def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
#     """Entropy of softmax distribution from logits."""
#     return -(x.softmax(1) * math.log2(math.e) * x.log_softmax(1)).sum(1)
