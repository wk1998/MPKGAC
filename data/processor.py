from curses import noecho
import os
from selectors import EpollSelector
import sys
import csv
import json
import torch
import pickle
import logging
import inspect
import contextlib
from tqdm import tqdm
from functools import partial
from collections import Counter
from multiprocessing import Pool
from collections import defaultdict
from dataclasses import dataclass, asdict
from torch.utils.data import Dataset
from transformers.models.auto.tokenization_auto import AutoTokenizer
import common_io
from random import sample
import random

logger = logging.getLogger(__name__)


def lmap(a, b):
    return list(map(a,b))



def cache_results(_cache_fp, _refresh=False, _verbose=1):
    def wrapper_(func):
        signature = inspect.signature(func)
        for key, _ in signature.parameters.items():
            if key in ('_cache_fp', '_refresh', '_verbose'):
                raise RuntimeError("The function decorated by cache_results cannot have keyword `{}`.".format(key))

        def wrapper(*args, **kwargs):
            my_args = args[0]
            mode = args[-1]
            if '_cache_fp' in kwargs:
                cache_filepath = kwargs.pop('_cache_fp')
                assert isinstance(cache_filepath, str), "_cache_fp can only be str."
            else:
                cache_filepath = _cache_fp
            if '_refresh' in kwargs:
                refresh = kwargs.pop('_refresh')
                assert isinstance(refresh, bool), "_refresh can only be bool."
            else:
                refresh = _refresh
            if '_verbose' in kwargs:
                verbose = kwargs.pop('_verbose')
                assert isinstance(verbose, int), "_verbose can only be integer."
            else:
                verbose = _verbose
            refresh_flag = True
            
            model_name = my_args.model_name_or_path.split("/")[-1]
            is_pretrain = my_args.pretrain
            cache_filepath = os.path.join(my_args.data_dir, f"cached_{mode}_features{model_name}_pretrain{is_pretrain}.pkl")
            refresh = my_args.overwrite_cache

            if cache_filepath is not None and refresh is False:
                # load data
                if os.path.exists(cache_filepath):
                    with open(cache_filepath, 'rb') as f:
                        results = pickle.load(f)
                    if verbose == 1:
                        logger.info("Read cache from {}.".format(cache_filepath))
                    refresh_flag = False

            if refresh_flag:
                results = func(*args, **kwargs)
                if cache_filepath is not None:
                    if results is None:
                        raise RuntimeError("The return value is None. Delete the decorator.")
                    with open(cache_filepath, 'wb') as f:
                        pickle.dump(results, f)
                    logger.info("Save cache to {}.".format(cache_filepath))

            return results

        return wrapper

    return wrapper_

# 随机遮掩一些文本输入
def random_mask_text_token(s,v_value):

    # 对所有样本，按照20%的概率遮掩掉预测答案v值
    if random.random()<0.2:
        if v_value[-1] in ["风","款","季","型"]:
            v_value=v_value[:len(v_value)-1]
        s = s.replace(v_value,'')
    return s

    # 对所有样本，按照10%的概率遮掩掉文本中15%的字
    # chars=[]
    # if random.random()<0.9:
    #     return s
    # else:
    #     for i in s:
    #         if random.random()<0.85:
    #             chars.append(i)
    #         else:
    #             chars.append('[MASK]')
    # return ''.join(chars)







def solve(line,  set_type="train", pretrain=1):
    examples = []
        
    head_ent_text = ent2text[line[0]]
    tail_ent_text = ent2text[line[2]]
    relation_text = rel2text[line[1]]
    
    i=0

    a = tail_filter_entities["\t".join([line[0],line[1]])]
    b = head_filter_entities["\t".join([line[2],line[1]])]
    
    guid = "%s-%s" % (set_type, i)
    text_a = head_ent_text
    # text_a= ''
    # text_b = relation_text.split("：")[0]+','.join(sample(relation_text.split("：")[1].split(','),5))
    text_b = "[SEP]"+relation_text.split("：")[0].replace('，比如','')
    # text_b = ''
    text_c = tail_ent_text
    v_value = tail_ent_text

    # 添加环境信息进去
    neibors=h2rt[line[0]]
    # print("neibors",len(neibors))

    if pretrain:
        # 带关系实体名字
        # text_c='[SEP]'.join([kv+' '+rel2token[kv.split(':')[0]]+" "+ent2token[kv.split(':')[1]] for kv in neibors])
        # 不带实体关系名字
        text_c='[SEP]'.join([rel2token[kv.split(':')[0]]+" "+ent2token[kv.split(':')[1]] for kv in neibors])
        # text_a=random_mask_text_token(text_a,'#')

        dic=defaultdict(list)
        for kv in neibors:
            k,v=kv.split(':')[0],kv.split(':')[1]
            dic[k].append(v)
        # print('neibors',neibors)
        # print("dic",dic)
        # print("line[0]",line[0])
        # 通过字符串长度来判断是商品还是属性值
        if len(line[0])>4:
            # 商品
            prompt='[MASK]的'
            for k,v in dic.items():
                prompt=prompt+(k+rel2token[k]+'是'+'、'.join([i+ent2token[i] for i in v])+';')
        else:
            # 属性值
            prompt="[MASK]是{}{}，它是{}{}的一种取值".format(line[0],ent2token[line[0]],v2p[line[0]],rel2token[v2p[line[0]]])

        examples.append(
            InputExample(guid=guid, text_a=prompt, text_b= text_a, text_c = '', label=ent2id[line[0]], real_label=ent2id[line[0]], en=0, rel=0, entity=line[0],neibors=text_c))
        # examples.append(
        #     InputExample(guid=guid, text_a=prompt, text_b=" ", text_c = '', label=ent2id[line[0]], real_label=ent2id[line[0]], en=0, rel=0, entity=line[0],neibors=text_c))

    else:
        # 带关系实体名字
        # text_c='[SEP]'.join([kv+' '+rel2token[kv.split(':')[0]]+" "+ent2token[kv.split(':')[1]] for kv in neibors if kv!= (line[1]+":"+line[2])])
        # 不带实体关系名字
        text_c='[SEP]'.join([rel2token[kv.split(':')[0]]+" "+ent2token[kv.split(':')[1]] for kv in neibors if kv!= (line[1]+":"+line[2])])
        # text_a=random_mask_text_token(text_a,v_value)
        dic=defaultdict(list)
        for kv in neibors:
            k,v=kv.split(':')[0],kv.split(':')[1]
            if v!=line[2]:
                dic[k].append(v)
        # 3
        prompt="[UNK]的[PAD]{}的取值可能是[MASK]".format(v2p[line[2]])
        # 1
        # prompt="[UNK]的[PAD]{}是[MASK]".format(v2p[line[2]])
        # 2
        # prompt="[MASK]可能是[UNK]的[PAD]{}".format(v2p[line[2]])
        # ,它的".format(v2p[line[2]])
        # for k,v in dic.items():
        #     prompt=prompt+(k+rel2token[k]+'是'+'、'.join([i+ent2token[i] for i in v])+';')

        examples.append(
            InputExample(guid=guid, text_a=prompt, text_b= text_a + text_b , text_c = line[2], label=lmap(lambda x: ent2id[x], a), real_label=ent2id[line[2]], en=ent2id[line[0]], rel=rel2id[line[1]], entity=line[0],neibors=text_c))       
        # examples.append(
            # InputExample(guid=guid, text_a=prompt, text_b= " " , text_c = line[2], label=lmap(lambda x: ent2id[x], a), real_label=ent2id[line[2]], en=ent2id[line[0]], rel=rel2id[line[1]], entity=line[0],neibors=text_c))       
    return examples


def filter_init(head, tail, t1,t2, ent2id_, ent2token_, rel2id_,h2rt_,rel2token_):
    global head_filter_entities
    global tail_filter_entities
    global ent2text
    global rel2text
    global ent2id
    global ent2token
    global rel2id
    global h2rt
    global rel2token
    global v2p

    head_filter_entities = head
    tail_filter_entities = tail
    ent2text =t1
    rel2text =t2
    ent2id = ent2id_
    ent2token = ent2token_
    rel2id = rel2id_
    h2rt=h2rt_
    rel2token=rel2token_
    v2p={}
    # "人群":
    a=["mm","乖乖女","人","代购","太太","女娃","女宝","女童","女装","娇小","孩子","宝宝","小个子","小女孩","年轻人","明星","甜辣妹","男女","男孩","胖mm","胖妹妹","胖美眉","辣妹"]
    # "图案":
    b=["个性字母印花","公主","几何","几何图案","卡通","印花图案","口袋","圆","圆点","复古","娃娃","字母","宝宝","小女孩","小熊","拼接","拼色","数字","日系","时尚","木耳","条纹","格子","植物","欧美","波点","爱心","碎花","简约","纯色","绣花","罗纹","胖妹妹","腰带","花朵","花色","荷叶","蝴蝶结","蝴蝶结款","螺纹","韩版"]
    # "工艺":
    c=["仿古","做旧","刺绣","印制","印染","印花","扎染","抽褶","拼接","提花","水洗","磨破","织花","褶皱","贴布绣","重工","针织","钉珠","镂空","镶钻"]
    # "款式":
    d=["3D型","Polo款","V口","V型","v字领","下摆","不对称","不规则","两件套","中款","中长款","修身型","假两件","内搭款","分叉","刺绣式","剪标款","加厚","包屁","半身式","单件","单排","单排扣","卫衣","印花","印花款","卷边","厚款","及膝款","双扣","双排","口袋款","吊带款","喇叭型","喇叭裤","圆领","均码","坎肩","垂坠感","垮裤","夏款","外搭","外穿款","多兜","多口袋","多扣","大号","大码","头套式","夹克","套装式","字母款","宽松型","宽松腰","小脚","带式","带钻","常规款","床罩式","开衩","开衫","开衫式","开衫款","开襟","开身衫","微喇","怀旧款","打结","扣式","抽绳","抽绳款","抽褶","拉链","拉链款","拼接","拼色","挂脖式","排扣","撕边","撞色","收腰款","新品","新型","新货","最新","有开叉","有领","木耳边","松紧","松紧腰","柔软型","格子","正肩","毛衣","毛边款","洞洞款","流苏款","版型","牛仔款","特宽型","瘦版","百搭款","百褶","直筒式","短款","短袖","矮腰","破洞款","碎花款","磨白","祺袍","秋款","立体裁剪","立体装饰","简单款","简易款","系带","紧身","纯色款","纽扣款","纽扣装饰","经典款","绑带","绣花","缕空","罩衫","美式","背带款","花边","荷叶边","蕾丝","蕾丝边","薄款","蝴蝶结款","螺口","补丁","衬衫式","裤装","褶皱","西装式","西装裤","西装领","贴布","超柔","超短版","过膝款","连体型","连帽","连衣","连衣裙","重磅","钉珠款","镂空","镶钻款","长款","长袖","阔型","阔腿","阔腿裤","雪纺衫","露肩","露背","露脐","露腰","颈挂","高端款","高腰","鱼尾型"]
    # "适用季节":
    e=["冬季","夏","夏季","春夏季","春季","秋冬季","秋季"]
    # "风格":
    f=["INS风","OL风","chic风","个性风","中国风","中性风","仙女风","仙气风","仿古风","休闲运动风","休闲风","优质风","优雅风","俏皮风","公主","公主风","典雅风","冷淡风","创意风","初恋风","原创设计风","古典风","古风","可爱风","可甜","可盐可甜","名媛风","哈伦风","商务风","嘻哈风","复古","复古风","奢华风","女神风","学院风","小清新风","小香风","少女风","怀旧风","性感风","慵懒风","文艺风","新中式风","日式风","日系","时尚韩版风","时尚风","极简风","森女风","欧式风","欧美风","民族风","气质风","法式风","淑女风","温柔风","港式风","潮流风","牛仔风","甜美风","甜酷风","田园风","盐系风","简约","简约风","精致风","纯欲风","美式","美式复古风","美式风","艺术风","街头","街头风","赫本风","轻奢风","轻熟风","通勤","通勤风","青春风","韩式","韩式风","韩版","高级感","高街风","高贵风"]
    
    for i in a:
        v2p[i]="人群"
    for i in b:
        v2p[i]="图案"
    for i in c:
        v2p[i]="工艺"
    for i in d:
        v2p[i]="款式"
    for i in e:
        v2p[i]="适用季节"
    for i in f:
        v2p[i]="风格"


def delete_init(ent2text_):
    global ent2text
    ent2text = ent2text_


def convert_examples_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def convert_examples_to_features(example, max_seq_length, mode, pretrain=1):
    """Loads a data file into a list of `InputBatch`s."""
    text_a = " ".join(example.text_a.split()[:128])
    text_b = " ".join(example.text_b.split()[:128])
    text_c = " ".join(example.text_c.split()[:128])
    
    if pretrain:
        input_text_a = text_a
        input_text_b = text_b
    else:
        input_text_a = tokenizer.sep_token.join([text_a, text_b])
        input_text_b = text_c
    

    inputs = tokenizer(
        input_text_a,
        input_text_b,
        truncation="longest_first",
        max_length=max_seq_length,
        padding="longest",
        add_special_tokens=True,
    )
    assert tokenizer.mask_token_id in inputs.input_ids, "mask token must in input"

    features = asdict(InputFeatures(input_ids=inputs["input_ids"],
                            attention_mask=inputs['attention_mask'],
                            labels=torch.tensor(example.label),
                            label=torch.tensor(example.real_label)
        )
    )
    return features


# @cache_results(_cache_fp="./dataset")
def get_dataset(args, processor, label_list, tokenizer, mode):

    assert mode in ["train", "dev", "test"], "mode must be in train dev test!"

    # use training data to construct the entity embedding
    if args.faiss_init and mode == "test" and not args.pretrain:
        mode = "train"
    else:
        pass

    if mode == "train":
        print("read train data")
        train_examples = processor.get_train_examples(args.data_dir)
    elif mode == "dev":
        train_examples = processor.get_dev_examples(args.data_dir)
    else:
        train_examples = processor.get_test_examples(args.data_dir)

    features = []
    encoder = MultiprocessingEncoder(tokenizer, args)
    encoder.initializer()
    print("convert examples to features")
    for i in range(len(train_examples)):
        if i%100==0:
            # print("#"*50)
            print("进度：",'%.2f'%(i/len(train_examples)))
        features.append(encoder.convert_examples_to_features(train_examples[i]))

    # with open(os.path.join(args.data_dir, f"examples_{mode}.txt"), 'w') as file:
    #     for line in train_examples:
    #         d = {}
    #         d.update(line.__dict__)
    #         file.write(json.dumps(d) + '\n')
    
    # features = []
    # # tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    # file_inputs = [os.path.join(args.data_dir, f"examples_{mode}.txt")]
    # file_outputs = [os.path.join(args.data_dir, f"features_{mode}.txt")]
    # with contextlib.ExitStack() as stack:
    #     inputs = [
    #         stack.enter_context(open(input, "r", encoding="utf-8"))
    #         if input != "-" else sys.stdin
    #         for input in file_inputs
    #     ]
    #     outputs = [
    #         stack.enter_context(open(output, "w", encoding="utf-8"))
    #         if output != "-" else sys.stdout
    #         for output in file_outputs
    #     ]

    #     encoder = MultiprocessingEncoder(tokenizer, args)
    #     pool = Pool(16, initializer=encoder.initializer)
    #     encoder.initializer()
    #     encoded_lines = pool.imap(encoder.encode_lines, zip(*inputs), 1000)
    #     # encoded_lines = map(encoder.encode_lines, zip(*inputs))
    #     # print(len(train_examples),train_examples[0])
    #     stats = Counter()
    #     for i, (filt, enc_lines) in tqdm(enumerate(encoded_lines, start=1), total=len(train_examples)):
    #         if filt == "PASS":
    #             for enc_line, output_h in zip(enc_lines, outputs):
    #                 features.append(eval(enc_line))
    #         else:
    #             stats["num_filtered_" + filt] += 1

    #     for k, v in stats.most_common():
    #         print("[{}] filtered {} lines".format(k, v), file=sys.stderr)

    num_entities = len(processor.get_entities(args.data_dir))
    print("num_entities:",num_entities)
    for f_id, f in enumerate(features):
        en = features[f_id].pop("en")
        rel = features[f_id].pop("rel")
        for i,t in enumerate(f['s_input_ids']):
            if t == tokenizer.unk_token_id:
                # features[f_id]['input_ids'][i] = en + len(tokenizer)
                features[f_id]['s_input_ids'][i] = en + 21128
                break
        
        for i,t in enumerate(f['s_input_ids']):
            if t == tokenizer.pad_token_id:
                # features[f_id]['input_ids'][i] = rel + len(tokenizer) + num_entities
                features[f_id]['s_input_ids'][i] = rel + 21128 + num_entities
                break

    features = KGCDataset(features)
    return features


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, real_label=None, en=None, rel=None, entity=None,neibors=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            text_c: (Optional) string. The untokenized text of the third sequence.
            Only must be specified for sequence triple tasks.
            label: (Optional) string. list of entities
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.real_label = real_label
        self.en = en
        self.rel = rel # rel id
        self.entity = entity
        # self.neibors = neibors


@dataclass
class InputFeatures:
    """A single set of features of data."""

    t_input_ids: torch.Tensor
    t_attention_mask: torch.Tensor

    s_input_ids: torch.Tensor
    s_attention_mask: torch.Tensor

    labels: torch.Tensor = None
    label: torch.Tensor = None

    en: torch.Tensor = 0
    rel: torch.Tensor = 0
    entity: torch.Tensor = None


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""

        with open(input_file,'r') as f:
            # print(input_file,len(lines))
            lines=f.readlines()
            cnt = len(lines)
            res=[]
            i=0
            for line in lines:
                # i+=1
                # if i>100:
                #     break
                line=line.split(",")
                if type(line[0])==int:
                    res.append([str(line[0]),str(line[1],'utf-8'),str(line[2],'utf-8')])
                else:
                    res.append([str(line[0],'utf-8'),str(line[1],'utf-8'),str(line[2],'utf-8')])
            return res



class KGProcessor(DataProcessor):
    """Processor for knowledge graph data set."""
    def __init__(self, tokenizer, args):
        self.labels = set()
        self.tokenizer = tokenizer
        self.args = args

        self.entity_path=self.args.tables.split(',')[0]
        self.relation_path=self.args.tables.split(',')[1]
        self.train_triples=self.args.tables.split(',')[2]
        self.vaild_triples=self.args.tables.split(',')[3]
        self.test_triples=self.args.tables.split(',')[4]


        # self.entity_path = os.path.join(args.data_dir, "entity2textlong2.txt") if os.path.exists(os.path.join(args.data_dir, 'entity2textlong2.txt')) \
        # else os.path.join(args.data_dir, "entity2text2.txt")

    
    def get_train_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", data_dir, self.args)
        return self._create_examples(
            self._read_tsv(self.train_triples),"train", data_dir, self.args)

    def get_dev_examples(self, data_dir):
        """See base class."""
        # return self._create_examples(
        #     self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", data_dir, self.args)
        return self._create_examples(
            self._read_tsv(self.vaild_triples), "dev", data_dir, self.args)

    def get_test_examples(self, data_dir, chunk=""):
      """See base class."""
    #   return self._create_examples(
    #       self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv")), "test", data_dir, self.args)
      return self._create_examples(
          self._read_tsv(self.test_triples), "test", data_dir, self.args)

    def get_relations(self, data_dir):
        """Gets all labels (relations) in the knowledge graph."""
        with open(self.relation_path,'r') as f:
            lines=f.readlines()
            relations = []
            for line in lines:
                line=line.split(',')
                relations.append(line[0])

        rel2token = {ent : f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        return list(rel2token.values())

    def get_labels(self, data_dir):
        """Gets all labels (0, 1) for triples in the knowledge graph."""
        relation = []
        with open(self.relation_path,'r') as f:
            lines=f.readlines()
            for line in lines:
                line=line.split(',')
                relation.append(line[0])

        return relation

    def get_entities(self, data_dir):
        """Gets all entities in the knowledge graph."""
        # with common_io.table.TableReader(self.entity_path) as f:
        with open(self.entity_path,'r') as f:
            ent_lines=f.readlines()
            entities = []
            for line in ent_lines:
                line=line.split(',')
                entities.append(line[0])
        
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        return list(ent2token.values())

    def get_train_triples(self, data_dir):
        """Gets training triples."""
        # return self._read_tsv(os.path.join(data_dir, "train.tsv"))
        
        return self._read_tsv(self.train_triples)

    def get_dev_triples(self, data_dir):
        """Gets validation triples."""
        # return self._read_tsv(os.path.join(data_dir, "dev.tsv"))
        return self._read_tsv(self.vaild_triples)

    def get_test_triples(self, data_dir, chunk=""):
        """Gets test triples."""
        # return self._read_tsv(os.path.join(data_dir, f"test{chunk}.tsv"))
        return self._read_tsv(self.test_triples)

    def _create_examples(self, lines, set_type, data_dir, args):
        """Creates examples for the training and dev sets."""
        # entity to text
        ent2text = {}
        ent2text_with_type = {}

        with open(self.entity_path,'r') as f:
            ent_lines=f.readlines()
            for line in ent_lines:
                line=line.split(',')
                if type(line[0])!=str:
                    ent2text[str(line[0],'utf-8')]=str(line[1],'utf-8')
                else:
                    ent2text[line[0]]=str(line[1],'utf-8')
  
        entities = list(ent2text.keys())
        ent2token = {ent : f"[ENTITY_{i}]" for i, ent in enumerate(entities)}
        ent2id = {ent : i for i, ent in enumerate(entities)}
        
        rel2text = {}

        with open(self.relation_path,'r') as f:
            ent_lines=f.readlines()
            for line in ent_lines:
                line=line.split(',')
                if type(line[0])!=str:
                    rel2text[str(line[0],'utf-8')]=str(line[1],'utf-8')
                else:
                    rel2text[line[0]]=str(line[1],'utf-8')


        relation_names = {}

        with open(self.relation_path,'r') as f:
            ent_lines=f.readlines()
            for line in ent_lines:
                line=line.split(',')
                relation_names[line[0]]=line[1]

        with open(self.relation_path,'r') as f:
            ent_lines=f.readlines()
            relations = []
            for line in ent_lines:
                line=line.split(',')
                relations.append(line[0])
        rel2token = {str(ent,'utf-8') : f"[RELATION_{i}]" for i, ent in enumerate(relations)}
        print('rel2token',rel2token)

        tmp_lines = []
        not_in_text = 0
        for line in tqdm(lines, desc="delete entities without text name."):
            if (line[0] not in ent2text) or (line[2] not in ent2text) or (line[1] not in rel2text):
                not_in_text += 1
                continue
            tmp_lines.append(line)
        lines = tmp_lines
        print(f"total entity not in text : {not_in_text} ")

        # rel id -> relation token id
        # rel2id = {w:i for i,w in enumerate(relation_names.keys())}
        rel2id = {w:i for i,w in enumerate(rel2text.keys())}

        examples = []
        # head filter head entity
        head_filter_entities = defaultdict(list)
        tail_filter_entities = defaultdict(list)

        # 记录邻居信息
        h2rt=defaultdict(list)

        # dataset_list = ["train.tsv", "dev.tsv", "test.tsv"]
        dataset_list = [self.train_triples, self.vaild_triples, self.test_triples]
        # in training, only use the train triples
        if set_type == "train" and not args.pretrain: dataset_list = dataset_list[0:1]

        for m in dataset_list:
            # with open(os.path.join(data_dir, m), 'r') as file:
            #     train_lines = file.readlines()
            #     for idx in range(len(train_lines)):
            #         train_lines[idx] = train_lines[idx].strip().split("\t")
            train_lines=self._read_tsv(m)

            for line in train_lines:
                # print('-'*90)
                # print(line[0],line[1],line[2])
                # print(type(line[0]),type(line[1]),type(line[2]))
                tail_filter_entities["\t".join([line[0], line[1]])].append(line[2])
                head_filter_entities["\t".join([line[2], line[1]])].append(line[0])

                h2rt[line[0]].append(':'.join([line[1],line[2]]))
        
        # print('h2rt',h2rt)
        max_head_entities = max(len(_) for _ in head_filter_entities.values())
        max_tail_entities = max(len(_) for _ in tail_filter_entities.values())
        max_h2rt=max(len(_) for _ in h2rt.values())


        # use bce loss, ignore the mlm
        if set_type == "train" and args.bce:
            # print("!"*90)
            lines = []
            for k, v in tail_filter_entities.items():
                h, r = k.split('\t')
                t = v[0]
                lines.append([h, r, t])
            for k, v in head_filter_entities.items():
                t, r = k.split('\t')
                h = v[0]
                lines.append([h, r, t])
        

        # for training , select each entity as for get mask embedding.
        if args.pretrain:
            # print("!")*90
            rel = list(rel2text.keys())[0]
            lines = []
            for k in ent2text.keys():
                # lines.append([k, rel, k])
                # print([k, rel, k])
                lines.append([k,rel,k])
        
        print(f"max number of filter entities : {max_head_entities} {max_tail_entities}")
        print(f"max h2rt: {max_h2rt}" )

        from os import cpu_count
        threads = min(1, cpu_count())
        filter_init(head_filter_entities, tail_filter_entities,ent2text, rel2text, ent2id, ent2token, rel2id, h2rt,rel2token
            )
        
        annotate_ = partial(
                solve,
                pretrain=self.args.pretrain
            )
        examples = list(
            tqdm(
                map(annotate_, lines),
                total=len(lines),
                desc="convert text to examples"
            )
        )

        tmp_examples = []
        for e in examples:
            for ee in e:
                tmp_examples.append(ee)
        examples = tmp_examples
        # delete vars
        del head_filter_entities, tail_filter_entities, ent2text, rel2text, ent2id, ent2token, rel2id

        print('examples:',len(examples))
        return examples


class Verbalizer(object):
    def __init__(self, args):
        if "WN18RR" in args.data_dir:
            self.mode = "WN18RR"
        elif "FB15k" in args.data_dir:
            self.mode = "FB15k"
        elif "umls" in args.data_dir:
            self.mode = "umls"
          
    def _convert(self, head, relation, tail):
        if self.mode == "umls":
            return f"The {relation} {head} is "
        
        return f"{head} {relation}"


class KGCDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __getitem__(self, index):
        return self.features[index]
    
    def __len__(self):
        return len(self.features)


class MultiprocessingEncoder(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.pretrain = args.pretrain
        self.max_seq_length = args.max_seq_length

    def initializer(self):
        global bpe
        bpe = self.tokenizer

    def encode(self, line):
        global bpe
        ids = bpe.encode(line)
        return list(map(str, ids))

    def decode(self, tokens):
        global bpe
        return bpe.decode(tokens)

    def encode_lines(self, lines):
        """
        Encode a set of lines. All lines will be encoded together.
        """
        enc_lines = []
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                return ["EMPTY", None]
            # print(type(line),line)
            enc_lines.append(json.dumps(self.convert_examples_to_features(example=eval(line))))
        return ["PASS", enc_lines]

    def decode_lines(self, lines):
        dec_lines = []
        for line in lines:
            tokens = map(int, line.strip().split())
            dec_lines.append(self.decode(tokens))
        return ["PASS", dec_lines]

    def convert_examples_to_features(self, example):
        pretrain = self.pretrain
        max_seq_length = self.max_seq_length
        global bpe
        """Loads a data file into a list of `InputBatch`s."""

        example1={}
        example1['text_a']=example.text_a
        example1['text_b']=example.text_b
        example1['text_c']=example.text_c
        example1['label']=example.label
        example1['real_label']=example.real_label
        example1['en']=example.en
        example1['rel']=example.rel
        example1['entity']=example.entity

        # 待预测对象
        text_a = example1['text_a'] 
        # 文本信息
        text_b = example1['text_b']
        # 环境信息
        text_c = example1['text_c']

        if pretrain:
            # the des of xxx is [MASK] .
            # xxx is the description of [MASK].
            # input_text = f"The description of {text_a} is that {text_b} ."
            t_input_text = text_b
            s_input_text = text_a

            # print("结构输入:",s_input_text)
            # print("文本输入:",t_input_text)
            # print("答案实体:",text_c)

            t_inputs = bpe(
                t_input_text,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )

            s_inputs = bpe(
                s_input_text,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )

        else:
            
            t_input_text = text_b

            # s_input_text = text_a+text_c
            s_input_text = text_a

            # print("结构输入:",s_input_text)
            # print("文本输入:",t_input_text)
            # print("答案实体:",text_c)

            t_inputs = bpe(
                t_input_text,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )

            s_inputs = bpe(
                s_input_text,
                truncation="longest_first",
                max_length=max_seq_length,
                padding="longest",
                add_special_tokens=True,
            )
        assert bpe.mask_token_id in s_inputs.input_ids, "mask token must in s_input"

        features = asdict(InputFeatures(
                                t_input_ids=t_inputs["input_ids"],
                                t_attention_mask=t_inputs['attention_mask'],

                                s_input_ids=s_inputs["input_ids"],
                                s_attention_mask=s_inputs['attention_mask'],

                                labels=example1['label'],
                                label=example1['real_label'],
                                en=example1['en'],
                                rel=example1['rel'],
                                entity=example1['entity']
            )
        )
        return features
