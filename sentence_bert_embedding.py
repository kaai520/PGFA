from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging
from pprint import pprint

def read_file(path):
    with open(path, 'r') as f:
        return f.read().split('\n')

def write_file(content,path):
    with open(path, 'w') as f:
        f.write('\n'.join(content))

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
def count_words(sentences_list):
    counter = 0
    for sentence in sentences_list:
        words = sentence.split()
        counter += len(words)
    return counter

# path = './descriptions/ntu_chatgpt_des.txt'
# des_list = read_file(path)
# max_len = 0
# label_list = []
# part_list = []
# for i, des in enumerate(des_list):
#     label, part_des = des.split(';',1)
#     label_list.append(label)
#     part_list.append(part_des)

# label_path = './descriptions/ntu_labelmap.txt'
# part_path = './descriptions/ntu_part_des.txt'
# write_file(label_list, label_path)
# write_file(part_list, part_path)
# path = './descriptions/ntu_parts_des.txt'
# path_des = './descriptions/ntu120_des.txt'

# path = './descriptions/ntu_parts_from_GAP.txt'
# part_list = read_file(path)
# des_list = [des.replace(';', '.') for des in des_list]
# des_part_list = part_list # []
# for i, part in enumerate(part_list):
#     label, part_des = part.split(';',1)
#     des_part_list.append(part_des)

# write_file(des_part_list, './descriptions/ntu_des+parts_des.txt')

# des_list = [des.replace(';', '.') for des in des_part_list]
# ntu_60_list = des_list[:60]

# path = './descriptions/ntu_labelmap.txt'
path = './descriptions/pku_des.txt'

des_list = read_file(path)
ntu_60_list = des_list[:60]
# pprint(ntu_60_list)


# Load pre-trained Sentence Transformer Model. It will be downloaded automatically
model = SentenceTransformer('all-mpnet-base-v2') 


sentence_embeddings = model.encode(des_list)
print(sentence_embeddings.shape)
save_path = './data/language/pku_des_embeddings.npy'
np.save(save_path, sentence_embeddings)