import struct
import torch
import torch.nn as nn
import numpy as np
from .base import BaseLitModel
from transformers.optimization import get_linear_schedule_with_warmup
from functools import partial
from .utils import LabelSmoothSoftmaxCEV1
from typing import Callable, Iterable, List
import common_io
from torch.utils.tensorboard import SummaryWriter
import os
import time

def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))

def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def decode(output_ids, tokenizer):
    return lmap(str.strip, tokenizer.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True))

class TransformerLitModel(BaseLitModel):
    def __init__(self, model, args, tokenizer=None, data_config={}):
        super().__init__(model, args)
        self.save_hyperparameters(args)
        if args.bce:
            self.loss_fn = nn.BCEWithLogitsLoss()
        elif args.label_smoothing != 0.0:
            self.loss_fn = LabelSmoothSoftmaxCEV1(lb_smooth=args.label_smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.best_acc = 0
        self.first = True
        self.tokenizer = tokenizer
        self.__dict__.update(data_config)
        self.args=args

        # resize the word embedding layer
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.decode = partial(decode, tokenizer=self.tokenizer)

        if args.pretrain:
            # when pretrain, only tune embedding layers
            self._freeze_attention()
        
        self.entity_path=args.tables.split(',')[0]
        self.token2ent=self.get_entities()

        # 生成log记录器和log目录

        self.writer = SummaryWriter(log_dir=args.log_path)
        self.t = 0
        self.e = 0
        self.loss_list=[]
        self.metric_list=[]

    def count_step(self):
        self.t+=1
        return self.t

    def conut_epoch(self):
        self.e+=1
        return self.e


    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        labels = batch.pop("labels")
        label = batch.pop("label")

        input_ids = batch['s_input_ids']
        logits = self.model(**batch,return_dict=False,pretrain=self.args.pretrain,struct_poss=self.args.struct_poss).logits

        # 取出mask位置的打分结果
        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        bs = input_ids.shape[0]
        mask_logits = logits[torch.arange(bs), mask_idx][:, self.entity_id_st:self.entity_id_ed]
        assert mask_idx.shape[0] == bs, "only one mask in sequence!"

        # finetune，二元交叉熵，多标签分类
        if self.args.bce:
            loss = self.loss_fn(mask_logits, labels)
        # pretrain，交叉熵，多分类
        else:
            loss = self.loss_fn(mask_logits, label)

        if batch_idx == 0:
            print('\n'.join(self.decode(batch['s_input_ids'][:4])))
            print('\n'.join(self.decode(batch['t_input_ids'][:4])))


        # 记录loss
        t=self.count_step()
        self.loss_list.append([t,loss])

        self.log("loss",loss,rank_zero_only=True)
        self.log("lr",self.lr,rank_zero_only=True)
        print("loss:",loss)
        print("lr:",self.lr)

        

        return loss

    def _eval(self, batch, batch_idx, ):
        labels = batch.pop("labels")
        input_ids = batch['s_input_ids']
        # single label
        label = batch.pop('label')  # bsz
        logits = self.model(**batch, return_dict=False,pretrain=self.args.pretrain).logits[:, :, self.entity_id_st:self.entity_id_ed] # bsz, len, entites

        _, mask_idx = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)    # bsz
        bsz = input_ids.shape[0]
        logits = logits[torch.arange(bsz), mask_idx] # bsz, entites


        # 过滤1对多实体之前的排名 ===========================================================
        print("logits.shape",logits.shape)
        value,index1=torch.topk(logits,20,dim=1,largest=True,sorted=True)
        b=index1
        _, outputs1 = torch.sort(logits, dim=1, descending=True) # bsz, entities   index
        # token id - 排名
        _, outputs1 = torch.sort(outputs1, dim=1)
        ranks1 = outputs1[torch.arange(bsz), label].detach().cpu() + 1


        # get the entity ranks
        # filter the entity
        assert labels[0][label[0]], "correct ids must in filiter!"
        labels[torch.arange(bsz), label] = 0
        assert logits.shape == labels.shape
        logits += labels * -100 # mask entity

        _, outputs = torch.sort(logits, dim=1, descending=True) # bsz, entities   index
        _, outputs = torch.sort(outputs, dim=1)
        ranks = outputs[torch.arange(bsz), label].detach().cpu() + 1

        # 过滤1对多后：打印测试实例输出 ===========================================================
        if batch_idx==0:
            value,index=torch.topk(logits,20,dim=1,largest=True,sorted=True)
            a=index
            # print("prob:",value)
            for i in range(a.shape[0]):
                print('-'*80)
                tmp=[]
                for j in a[i][:]:
                    j=j+21128
                    ent=self.token2ent[self.tokenizer.convert_ids_to_tokens(int(j))]
                    if type(ent)!=str:
                        ent=str(ent,'utf-8')
                    tmp.append(ent)
                answer=self.tokenizer.convert_ids_to_tokens(label+21128)[i]
                tmp1=[]
                for j in b[i][:]:
                    j=j+21128
                    ent=self.token2ent[self.tokenizer.convert_ids_to_tokens(int(j))]
                    if type(ent)!=str:
                        ent=str(ent,'utf-8')
                    tmp1.append(ent)

                print("预测样例:")
                print(''.join(self.tokenizer.batch_decode(batch['t_input_ids'][i])).replace(' ',''))
                print(''.join(self.tokenizer.batch_decode(batch['s_input_ids'][i])).replace(' ',''))
                
                print("过滤前:")
                print("Ground truth 排名:",label.detach().cpu()[i],"|",ranks1[i],"|",answer,str(self.token2ent[answer],'utf-8'))
                print("排名前20的预测实体:",' | '.join(tmp1))

                print("过滤后:")
                print("Ground truth 排名:",label.detach().cpu()[i],"|",ranks[i],"|",answer,str(self.token2ent[answer],'utf-8'))
                print("排名前20的预测实体:",' | '.join(tmp))
        # =========================================================================

        return dict(ranks = np.array(ranks))

    def get_entities(self):
        """Gets all entities in the knowledge graph."""

        with common_io.table.TableReader(self.entity_path) as f:
            cnt = f.get_row_count()
            ent_lines=f.read(cnt)
            entities = []
            for line in ent_lines:
                entities.append(line[0])
        
        token2ent = {f"[ENTITY_{i}]":ent  for i, ent in enumerate(entities)}
        return token2ent

    def validation_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        return result

    def validation_epoch_end(self, outputs) -> None:
        ranks = np.concatenate([_['ranks'] for _ in outputs])
        total_ranks = ranks.shape[0]

        if not self.args.pretrain:
            l_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2)))]
            r_ranks = ranks[np.array(list(np.arange(0, total_ranks, 2))) + 1]
            self.log("Eval/lhits10", (l_ranks<=10).mean(),rank_zero_only=True)
            self.log("Eval/rhits10", (r_ranks<=10).mean(),rank_zero_only=True)

        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

        self.log("Eval/hits10", hits10,rank_zero_only=True)
        self.log("Eval/hits20", hits20,rank_zero_only=True)
        self.log("Eval/hits3", hits3,rank_zero_only=True)
        self.log("Eval/hits1", hits1,rank_zero_only=True)
        self.log("Eval/mean_rank", ranks.mean(),rank_zero_only=True)
        self.log("Eval/mrr", (1. / ranks).mean(),rank_zero_only=True)
        self.log("hits10", hits10, prog_bar=True,rank_zero_only=True)
        self.log("hits1", hits1, prog_bar=True,rank_zero_only=True)
        e=self.conut_epoch()
        self.metric_list.append([hits10,hits20,hits3,hits1, ranks.mean(),(1. / ranks).mean(),e])
  

    def test_step(self, batch, batch_idx):
        result = self._eval(batch, batch_idx)
        # self.log("Test/ranks", np.mean(ranks))


        return result

    def test_epoch_end(self, outputs) -> None:
        print("记录loss！")
        with open(self.args.log_path+"loss.txt",'a+') as f:
            for i in range(len(self.loss_list)):
                t=str(self.loss_list[i][0])
                loss=str(self.loss_list[i][1].item())
                print(t+","+loss+"\n")
                f.write(t+","+loss+"\n")
        print("已完成记录loss！")

        print("记录metric！")
        with open(self.args.log_path+"metric.txt",'a+') as f:
            for i in range(len(self.metric_list)):
                hit10=str(self.metric_list[i][0])
                hit20=str(self.metric_list[i][1])
                hit3=str(self.metric_list[i][2])
                hit1=str(self.metric_list[i][3])
                mr=str(self.metric_list[i][4])
                mrr=str(self.metric_list[i][5])
                epoch=str(self.metric_list[i][6])
                print(hit10+","+hit20+","+hit3+","+hit1+","+mr+","+mrr+","+epoch+"\n")
                f.write(hit10+","+hit20+","+hit3+","+hit1+","+mr+","+mrr+","+epoch+"\n")
        print("已完成记录metric！")    



        ranks = np.concatenate([_['ranks'] for _ in outputs])
        hits20 = (ranks<=20).mean()
        hits10 = (ranks<=10).mean()
        hits3 = (ranks<=3).mean()
        hits1 = (ranks<=1).mean()

       
        self.log("Test/hits10", hits10,rank_zero_only=True)
        self.log("Test/hits20", hits20,rank_zero_only=True)
        self.log("Test/hits3", hits3,rank_zero_only=True)
        self.log("Test/hits1", hits1,rank_zero_only=True)
        self.log("Test/mean_rank", ranks.mean(),rank_zero_only=True)
        self.log("Test/mrr", (1. / ranks).mean(),rank_zero_only=True)

    def configure_optimizers(self):
        no_decay_param = ["bias", "LayerNorm.weight"]

        optimizer_group_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay_param)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay_param)], "weight_decay": 0}
        ]

        optimizer = self.optimizer_class(optimizer_group_parameters, lr=self.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_training_steps * self.args.warm_up_radio, num_training_steps=self.num_training_steps)
        return {
            "optimizer": optimizer, 
            "lr_scheduler":{
                'scheduler': scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1,
            }
        }
    
    def _freeze_attention(self):
        for k, v in self.model.named_parameters():
            # 预训练，放开word embedding和解码层参数
            if "word" not in k and 'prediction' not in k :
                v.requires_grad = False
            else:
                print(k)
    
    def _freaze_word_embedding(self):
        for k, v in self.model.named_parameters():
            if "word" in k:
                print(k)
                v.requires_grad = False

    @staticmethod
    def add_to_argparse(parser):
        parser = BaseLitModel.add_to_argparse(parser)

        parser.add_argument("--label_smoothing", type=float, default=0.1, help="")
        parser.add_argument("--bce", type=int, default=0, help="")
        return parser
