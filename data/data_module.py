import os
import torch
import random
import transformers
from PIL import Image
from enum import Enum
from os import listdir
from dataclasses import dataclass
from typing import Any, Optional, Union


from collections import defaultdict
import common_io
import io
import base64

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BertTokenizer
from transformers.models.clip import CLIPProcessor
from transformers.tokenization_utils_base import (BatchEncoding,
                                                  PreTrainedTokenizerBase)
from .base_data_module import BaseDataModule
from .processor import KGProcessor, get_dataset

transformers.logging.set_verbosity_error()

root_path="./"
aux_size, rcnn_size = 128, 64
clip_processor = CLIPProcessor.from_pretrained(root_path+'pretrain/clip-vit-base-patch32')
# 包装了图片CLIPfeatureExtractor和文本tokenizer
aux_processor = CLIPProcessor.from_pretrained(root_path+'pretrain/clip-vit-base-patch32')
# crop_size：输出维度默认224，size：Resize the input to the given size
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = aux_size, aux_size
rcnn_processor = CLIPProcessor.from_pretrained(root_path+'pretrain/clip-vit-base-patch32')
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = rcnn_size, rcnn_size


class ExplicitEnum(Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the ``padding`` argument in :meth:`PreTrainedTokenizerBase.__call__`. Useful for tab-completion
    in an IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    num_labels: int = 0
    task_name: str = None
    entity_img_path: str = None
    entity_img_files: Optional[Any] = None
    entity2img: Optional[Any] = None

    def __call__(self, features, return_tensors=None):

        if return_tensors is None:
            return_tensors = self.return_tensors
        # print("features+++",len(features),features[0].keys())

        labels = [feature.pop("labels") for feature in features] if "labels" in features[0].keys() else None
        label = [feature.pop("label") for feature in features]
        entities = [feature.pop("entity") for feature in features] if "entity" in features[0].keys() else None

        features_keys = {}
        for k in list(features[0].keys()):
            # ignore the padding arguments
            if k in ["s_input_ids", "s_attention_mask", "s_token_type_ids","t_input_ids", "t_attention_mask", "t_token_type_ids"]: continue
            features_keys[k] = [feature.pop(k) for feature in features]

        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        bsz = len(labels)
        with torch.no_grad():
            new_labels = torch.zeros(bsz, self.num_labels)
            for i,l in enumerate(labels):
                if isinstance(l, int): 
                    new_labels[i][l] = 1
                else:
                    for j in l:
                        new_labels[i][j] = 1
            labels = new_labels

        # print("-"*90)
        # print("features",len(features),features[0].keys())
        # print("self.padding",self.padding)
        # print("self.max_length",self.max_length)
        # print("self.pad_to_multiple_of",self.pad_to_multiple_of)
        # print("return_tensors",return_tensors)

        t_features=[]
        s_features=[]

        for feature in features:
            t_features.append({k[2:]:v for k,v in feature.items() if k=="t_input_ids" or k=='t_attention_mask'})
            s_features.append({k[2:]:v for k,v in feature.items() if k=="s_input_ids" or k=='s_attention_mask'})

        t_features = self.tokenizer.pad(
            t_features,
            padding=self.padding,
            max_length=32,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        s_features = self.tokenizer.pad(
            s_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        # print(len(s_features),len(t_features))
        # print(type(s_features),type(t_features))
        # print('s_feature',s_features)
        # print('t_feature',t_features)
        # print(type(features),type(features[i]))

        features={}
        features['t_input_ids']=t_features['input_ids']
        features['t_attention_mask']=t_features['attention_mask']
        features['s_input_ids']=s_features['input_ids']
        features['s_attention_mask']=s_features['attention_mask']

        features['labels'] = labels
        features['label'] = torch.tensor(label)

        features.update(features_keys)

        # region
        pixel_images, aux_images, rcnn_images = [], [], []
        # print(entities)
        for entity in entities:
            if self.task_name == 'wn18':
                en_file = 'n' + entity    # wn18
            elif self.task_name == 'fb15k-237':
                en_file = entity[1:].replace('/', '.') # m.01rng // n01443537
            elif self.task_name == '1688':
                # 包含base64的list
                en_file= self.entity2img[entity]
                # print("在读图片！",entity,type(entity))
                # print("图片数量",len(en_file))
            else:
                raise ValueError(
                f"{self.task_name} is not a valid task name, please select one of [wn18, fb15k-237,1688]"
            )
            en_imgs = []
            # if en_file in self.entity_img_files:
            if 1:

                # en_file = os.path.join(self.entity_img_path, en_file)
                # en_imgs = [os.path.join(en_file, file) for file in os.listdir(en_file)]

                # 把list当中的每一个base64转成图片
                # print("base64-2",type(en_file),len(en_file),en_file)
                en_imgs=[io.BytesIO(base64.b64decode(img)) for img in en_file]

                if len(en_imgs) > 7:    # random select six imgs
                    random.seed(1)
                    en_imgs = random.sample(en_imgs, k=7)

            en_full_imgs = en_imgs[:1]
            en_aux_imgs = en_imgs[1:4]
            en_rcnn_imgs = en_imgs[4:]

            if len(en_full_imgs) > 0:
                # print(en_full_imgs[0])
                try:
                    full_img = Image.open(en_full_imgs[0]).convert('RGB')
                    full_img = clip_processor(images=full_img, return_tensors='pt')['pixel_values'].squeeze()
                    pixel_images.append(full_img)
                except:
                    print("该图片解析有问题！",entity,en_full_imgs[0])
                    print("--------",len(pixel_images))
                    pixel_images.append(torch.zeros((3, 224, 224)))
                    print("--------",len(pixel_images))
                    
                    continue
                # print("####")
            else:
                # print("$$$$")
                pixel_images.append(torch.zeros((3, 224, 224)))
            
            aux_imgs, rcnn_imgs = [], []
            # select 3 imgs
            for i in range(min(3, len(en_aux_imgs))):
                aux_img = Image.open(en_aux_imgs[i]).convert('RGB')
                aux_img = aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
                aux_imgs.append(aux_img)
            for i in range(min(3, len(en_rcnn_imgs))):
                rcnn_img = Image.open(en_rcnn_imgs[i]).convert('RGB')
                rcnn_img = rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
                rcnn_imgs.append(rcnn_img)
            # padding
            for i in range(3-len(en_aux_imgs)):
                aux_imgs.append(torch.zeros((3, aux_size, aux_size))) 
            for i in range(3-len(en_rcnn_imgs)):
                rcnn_imgs.append(torch.zeros((3, rcnn_size, rcnn_size)))
            aux_images.append(torch.stack(aux_imgs))
            rcnn_images.append(torch.stack(rcnn_imgs))

        features['pixel_values'] = torch.stack(pixel_images)
        features['aux_values'] = torch.stack(aux_images)
        features['rcnn_values'] = torch.stack(rcnn_images)

        # print('pixel_values',features['pixel_values'].shape,entities)
        # print('aux_values',features['aux_values'].shape,entities)
        # print('rcnn_values',features['rcnn_values'].shape,entities)
        
        #endregion
        return features


class KGC(BaseDataModule):
    def __init__(self, args, model) -> None:
        super().__init__(args)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_name_or_path, use_fast=False)
        self.processor = KGProcessor(self.tokenizer, args)

        self.label_list = self.processor.get_labels(args.data_dir)
        entity_list = self.processor.get_entities(args.data_dir)

        print("实体数量：",len(entity_list))
        
        # 把实体作为token加入到词表当中
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': entity_list})
        print("实体作为token加入词表！",num_added_tokens)
        print("目前词表长度：",len(self.tokenizer))
        
        entity_img_path = {'wn18': 'dataset/wn18-images/', 'fb15k-237': root_path+'pretrain/dataset/FB15k-images/','1688':args.tables.split(',')[0]}[self.args.task_name]
        entity_img_files=''

        # 创建实体-图片字典
        self.entity2img=defaultdict(list)
        with open(args.tables.split(',')[0],'r') as f:
            ent_lines=f.readlines()
            img_num_sum=0
            for line in ent_lines:
                base64_img=str(line[2],'utf-8')
                if len(base64_img)<100:#v值赋空图片
                    base64_img="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"
                base64_imgs=[base64_img]*10
                self.entity2img[str(line[0],'utf-8')]=base64_imgs
                img_num_sum+=len(base64_imgs)

        print("实体平均图片数量：",img_num_sum/len(entity_list))

        # 对dataset的包装器，再处理函数
        self.sampler = DataCollatorForSeq2Seq(self.tokenizer,
            model=model,
            label_pad_token_id=self.tokenizer.pad_token_id,
            pad_to_multiple_of=8 if self.args.precision == 16 else None,
            padding="longest",
            max_length=self.args.max_seq_length,
            num_labels=len(entity_list),
            task_name=self.args.task_name,
            entity_img_path=entity_img_path,
            entity_img_files=entity_img_files,
            entity2img=self.entity2img

        )
        relations_tokens = self.processor.get_relations(args.data_dir)
        self.num_relations = len(relations_tokens)

        # 把关系作为token加入到词表当中
        num_added_tokens = self.tokenizer.add_special_tokens({'additional_special_tokens': relations_tokens})
        print("关系作为token加入词表！",num_added_tokens)
        print("目前词表长度：",len(self.tokenizer))

        # 获取实体关系的id范围
        vocab = self.tokenizer.get_added_vocab()    # dict: word: idx
        self.relation_id_st = vocab[relations_tokens[0]]
        self.relation_id_ed = vocab[relations_tokens[-1]] + 1
        self.entity_id_st = vocab[entity_list[0]]
        self.entity_id_ed = vocab[entity_list[-1]] + 1


    def setup(self, stage=None):
        print("生成训练数据！")
        self.data_train = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "train")
        self.data_val = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "dev") if not self.args.pretrain else self.data_train
        self.data_test = get_dataset(self.args, self.processor, self.label_list, self.tokenizer, "test") if not self.args.pretrain else self.data_train

    def prepare_data(self):
        pass

    def get_config(self):
        d = {}
        for k, v in self.__dict__.items():
            if "st" in k or "ed" in k:
                d.update({k:v})
        
        return d

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--model_name_or_path", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--data_dir", type=str, default="roberta-base", help="the name or the path to the pretrained model")
        parser.add_argument("--max_seq_length", type=int, default=256, help="Number of examples to operate on per forward step.")
        parser.add_argument("--warm_up_radio", type=float, default=0.1, help="Number of examples to operate on per forward step.")
        parser.add_argument("--eval_batch_size", type=int, default=8)
        parser.add_argument("--overwrite_cache", action="store_true", default=False)
        return parser

    def get_tokenizer(self):
        return self.tokenizer

    def train_dataloader(self):
        print("len(DataLoader)",len(DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=self.args.pretrain)))
        return DataLoader(self.data_train, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=self.num_workers, pin_memory=False, collate_fn=self.sampler, batch_size=self.args.eval_batch_size)

