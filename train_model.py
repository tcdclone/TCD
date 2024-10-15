import os
import argparse
from transformers import TrainingArguments, AutoModel, AutoTokenizer

from freeze_param import freeze_decoder_except_xattn_codegen
from clone_dataset import load_dataset, collate_fn
from TCDTrainer import TCDTrainer

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO']='0.0'
os.environ['TOKENIZERS_PARALLELISM']='true'

# args 
parser = argparse.ArgumentParser(description="CodeT5+ finetuning on Seq2Seq LM task")
parser.add_argument('--epochs', default=4, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--lr-warmup-steps', default=200, type=int)
parser.add_argument('--batch-size-per-replica', default=8, type=int)
parser.add_argument('--grad-acc-steps', default=1, type=int)
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--fp16', default=True, action='store_true')
parser.add_argument('--train_data', default='./dataset/fun_train.pkl', type=str)
parser.add_argument('--test_data', default='./dataset/fun_test.pkl', type=str)
parser.add_argument('--lambda', default=0.4, type=float)

args = parser.parse_args()

# loading model and dataset
model_name = "Salesforce/codet5p-220m-bimodal"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
freeze_decoder_except_xattn_codegen(model)

print("Loading model successed!")

train_dataset = load_dataset(args.train_data, tokenizer)
eval_dataset = load_dataset(args.test_data, tokenizer)

print("Loading Dataset Successed!")
print("Training Dataset len: ", len(train_dataset))
print("Test Dataset len: ", len(eval_dataset))

# setting training args and trainer
training_args = TrainingArguments(
    do_eval=True,
    do_train=True,

    # train
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size_per_replica,
    gradient_accumulation_steps=args.grad_acc_steps,
    learning_rate=args.lr,
    weight_decay=0.05,
    warmup_steps=args.lr_warmup_steps,
    local_rank=args.local_rank,
    fp16=args.fp16,

    # eval
    evaluation_strategy="steps",    
    eval_steps=500,
    per_device_eval_batch_size=args.batch_size_per_replica,
    eval_on_start=True,

    # log
    output_dir='./result',     
    save_total_limit=10,          
    logging_dir='./log',        
    logging_steps=10,           
    remove_unused_columns=False,

    dataloader_drop_last=True,
    dataloader_num_workers=10,        
)

trainer = TCDTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collate_fn,
    eval_dataset=eval_dataset,
)

# train the model
trainer.train()
