import torch
import torch.nn as nn
from pytorch_pretrained import BertTokenizer, BertModel


class Config(object):
    """
    config param
    """
    def __init__(self,dataset):
        self.model_name='BruceBert'
        #train set
        self.train_path = dataset + '/data/train.txt'
        #test set
        self.test_path = dataset + '/data/test.txt'
        # validation set
        self.dev_path = dataset + '/data/dev.txt'
        # 类别
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt').readlines()]
        # 模型训练结果
        # self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.pth'

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 若超过 1000 batch 还没有提升，就early stop
        self.require_improvement = 1000
        # 类别数
        self.num_classes = len(self.class_list)
        # epoch 数量
        self.num_epochs = 3
        # batch_size
        self.batch_size = 128
        # padding size
        self.pad_size = 32
        # learning rate
        self.learning_rate = 1e-5
        # bert 预训练的位置
        self.bert_path = 'bert_pretrain'
        # bert 切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        # bert 隐藏层
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x [ids, seq_len, mask]
        context = x[0] # 输入的句子 shape[batch_size,seq_len ] = [128,32]
        mask = x[2] # 对padding 部分进行mask
        _,pooled = self.bert(context, attention_mask = mask, output_all_encoded_layers = False)
        # pooled shape [128, 768]
        out = self.fc(pooled) # shape [128,10]
        return out

