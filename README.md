# MMPAC

是一种面向电商领域的多模态产品数据，图片，title以及原生属性三元组的多模态产品属性补全方法。采取三流架构对每种模态进行建模，并在融合层和解码层进行多模态信息的充分利用来增强属性补全效果。
It is a multi-modal product attribute completion method for multi-modal product data, pictures, titles and original attribute triples in the e-commerce field. A three-stream architecture is adopted to model each modality, and multi-modal information is fully utilized in the fusion layer and decoding layer to enhance the attribute completion effect.

---

## Dataset

- entity.txt：实体表，包含商品和属性值实体，每行格式为“entity id, title文本 ,图片base64”
- reation.txt：属性表，每行格式为“relation id, 属性解释文本”
- train/vaild/test.txt：三元组表，每行格式为“头实体id，尾实体if，关系id”；
- as shown in Demo dataset.

### Pretrain

- run: main.py
- config=“--gpus=1 --max_epochs=13  --num_workers=4 --model_name_or_path=bert-base-chinese --accumulate_grad_batches=1 --model_class=UnimoKGC --batch_size=64 --pretrain=1 --bce=0 --check_val_every_n_epoch=1 --data_dir==dataset --task_name=1688 --eval_batch_size=64 --max_seq_length=64 --lr=5e-4 --log_path=training/logs --tables=entity.txt,reation.txt,train.txt,vaild.txt,test.txt"

---

### Fintune

- run: main.py
- config=“--gpus=1 --max_epochs=12  --num_workers=0 --model_name_or_path=bert-base-chinese --accumulate_grad_batches=1 --model_class=UnimoKGC  --batch_size=48 --label_smoothing=0.3 --pretrain=0 --bce=1 --check_val_every_n_epoch=1 --data_dir=dataset --task_name=1688 --overwrite_cache --eval_batch_size=32 --max_seq_length=64 --lr=6e-5 --checkpoint=epoch=12-step=20760-Eval/hits10=1.00.ckpt --log_path=training/logs --tables=entity.txt,reation.txt,train.txt,vaild.txt,test.txt ”
