import torch
import wandb
import evaluate
import os 
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from betty.engine import Engine
from betty.problems import ImplicitProblem
from betty.configs import Config, EngineConfig
from utils import get_data_loader,argument_parser,get_data_loader_val,ModelArguments, DataTrainingArguments, OurTrainingArguments,task_to_keys
from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict, Tuple
from datasets import load_dataset, Dataset
from transformers import CONFIG_MAPPING,AutoConfig,AutoTokenizer,HfArgumentParser,default_data_collator,get_linear_schedule_with_warmup

from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from simcse.models import DebertaForCL, DebertaClassificationHead
from copy import deepcopy
from random import random as rand

parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))

parser = argument_parser(parser)
model_args, data_args, training_args, args = parser.parse_args_into_dataclasses()

print(args)
print(model_args)
print(data_args)
print(training_args)

if args.wandb:
    wandb.login()
    run = wandb.init(
        # Set the project where this run will be logged
        project="TapWeight",
        config={
            "learning_rate_reweight": args.reweight_lr,
            'step_size': args.step_size, 
            'task': args.task
        },
    )

if args.same_dataset:
    if args.task == 'imdb': 
        datasets_pt = load_dataset("stanfordnlp/imdb",split='train')
    elif args.task == 'agnews': 
        datasets_pt = load_dataset("fancyzhx/ag_news", split='train')
    elif args.task == 'scierc': 
        datasets_pt = load_dataset("hrithikpiyush/scierc", split='train')
    elif args.task == 'rct': 
        datasets_pt = load_dataset("armanc/pubmed-rct20k", split='train')
    else:
        datasets_pt = load_dataset("glue", args.task,split='train')
    data_list=[]
    sentence1_key, sentence2_key = task_to_keys[args.task]
    for d in datasets_pt:
        data_list.append({'text':d[sentence1_key]})
        if sentence2_key:
            data_list.append({'text':d[sentence2_key]})
    datasets={'train':Dataset.from_list(data_list)}
else:

    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/", delimiter="\t" if "tsv" in data_args.train_file else ",")
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir="./data/")


# See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
# https://huggingface.co/docs/datasets/loading_datasets.html.

# Load pretrained model and tokenizer
#
# Distributed training:
# The .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config_kwargs = {
    "cache_dir": model_args.cache_dir,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.config_name:
    config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
elif model_args.model_name_or_path:
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
else:
    config = CONFIG_MAPPING[model_args.model_type]()
    logger.warning("You are instantiating a new config instance from scratch.")

if args.task == 'stsb':
    config.num_labels=1
elif args.task == 'mnli':
    config.num_labels=3
elif args.task == 'agnews':
    config.num_labels=4
elif args.task == 'scierc':
    config.num_labels=7
elif args.task == 'rct':
    config.num_labels=5
else:
    config.num_labels=2

tokenizer_kwargs = {
    "cache_dir": model_args.cache_dir,
    "use_fast": model_args.use_fast_tokenizer,
    "revision": model_args.model_revision,
    "use_auth_token": True if model_args.use_auth_token else None,
}
if model_args.tokenizer_name:
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
elif model_args.model_name_or_path:
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
else:
    raise ValueError(
        "You are instantiating a new tokenizer from scratch. This is not supported by this script."
        "You can do it from another script, save it, and load it from here, using --tokenizer_name."
    )
model = DebertaForCL.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            model_args=model_args                  
        )

model.resize_token_embeddings(len(tokenizer))

# Prepare features
column_names = datasets["train"].column_names
sent2_cname = None
if len(column_names) == 2:
    # Pair datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
elif len(column_names) == 3:
    # Pair datasets with hard negatives
    sent0_cname = column_names[0]
    sent1_cname = column_names[1]
    sent2_cname = column_names[2]
elif len(column_names) == 1:
    # Unsupervised datasets
    sent0_cname = column_names[0]
    sent1_cname = column_names[0]
else:
    raise NotImplementedError

def prepare_features(examples):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the 
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    total = len(examples[sent0_cname])
    #print(total,sent0_cname)
    # Avoid "None" fields 
    for idx in range(total):
        if examples[sent0_cname][idx] is None:
            examples[sent0_cname][idx] = " "
        if examples[sent1_cname][idx] is None:
            examples[sent1_cname][idx] = " "
    
    sentences = examples[sent0_cname] + examples[sent1_cname]

    # If hard negative exists
    if sent2_cname is not None:
        for idx in range(total):
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

    sent_features = tokenizer(
        sentences,
        max_length=data_args.max_seq_length,
        truncation=True,
        padding="max_length" if data_args.pad_to_max_length else False,
    )
    
    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i+total], sent_features[key][i+total*2]] for i in range(total)]
    else:
        for key in sent_features:

            features[key] = [[sent_features[key][i], sent_features[key][i+total]] for i in range(total)]
        
    if model_args.do_sop:
        sentences_sop = examples[sent0_cname]

        features['sop_input_ids'],features['sop_labels']=[],[]
        for s_sop in sentences_sop:
            tokens_sop = tokenizer.tokenize(s_sop)
            len_sen_sop = min(len(tokens_sop)-1,data_args.max_seq_length-3)

            len_sen_sop_a = len_sen_sop // 2 
            tokens_a = tokens_sop[:len_sen_sop_a]
            tokens_b = tokens_sop[len_sen_sop_a:len_sen_sop]

            is_next = rand() < 0.5 # whether token_b is next to token_a or not
            if is_next:
                features['sop_input_ids'].append(tokenizer.convert_tokens_to_ids(['<s>'] + tokens_a + ['</s>'] + tokens_b + ['</s>']))
                features['sop_labels'].append([1])
            else:
                features['sop_input_ids'].append(tokenizer.convert_tokens_to_ids(['<s>'] + tokens_b + ['</s>'] + tokens_a + ['</s>']))
                features['sop_labels'].append([0])
    return features

if training_args.do_train:
    train_dataset = datasets["train"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

# Data collator
@dataclass
class OurDataCollatorWithPadding:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    mlm: bool = True
    mlm_probability: float = data_args.mlm_probability

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        #print(features[0])
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                #flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in ['input_ids', 'attention_mask']})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if model_args.do_sop:
            sop_features = []
            for feature in features:
                #print(feature)
                sop_features.append({'input_ids':feature['sop_input_ids'],'sop_labels':feature['sop_labels']})
            batch_sop = self.tokenizer.pad(
                sop_features,
                padding=self.padding,
                max_length=data_args.max_seq_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )
        

        if model_args.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        if model_args.do_sop:
            batch['sop_input_ids']=batch_sop['input_ids']
            batch['sop_labels']=batch_sop['sop_labels']
            batch['sop_attention_mask']=batch_sop['attention_mask']
        return batch
    
    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

#data_collator = default_data_collator if data_args.pad_to_max_length else OurDataCollatorWithPadding(tokenizer)
data_collator = OurDataCollatorWithPadding(tokenizer)


device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


finetune_dataloader,eval_dataloader=get_data_loader(args,config)
pretrain_dataloader = DataLoader(train_dataset, shuffle=False, collate_fn=data_collator,
                                    batch_size=args.batch_size,drop_last=True)

'''
model.to(device)
for batch in pretrain_dataloader:
    for k in batch:
        batch[k] = batch[k].to(device)
    outs = model(**batch)
    #print(outs)
    outs.loss.backward()
    exit()
'''

class Finetune(torch.nn.Module):
    def __init__(self):
        super(Finetune, self).__init__()
        self.backbone=deepcopy(model)
        self.head=DebertaClassificationHead(config).to(device).train()

    def forward(self,batch):
        outputs = self.backbone(**batch, output_hidden_states=True, return_dict=True,sent_emb=True)

        logits = self.head(outputs.pooler_output.unsqueeze(1))

        return logits

model_ft=Finetune()

class Reweight(torch.nn.Module):
    def __init__(self):
        super(Reweight, self).__init__()
        self.weight=torch.nn.Parameter(torch.ones(3))

    def forward(self):
        return torch.softmax(self.weight,0)

model_reweight = Reweight().to(device)

if args.task == 'stsb':
    loss_fct = MSELoss()
else:
    loss_fct = CrossEntropyLoss()

save_dir='{}_save'.format(args.task)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

class Pretraining(ImplicitProblem):
    def training_step(self, batch):
        #print('pretraining')
        for k in batch:
            batch[k] = batch[k].to(device)
        outs = self.module(**batch, mlm_weights=self.reweight.module())
        if args.wandb:
            wandb.log({"pretraining loss": outs.loss.item()})
            wandb.log({'pretraining lr': self.optimizer.param_groups[0]['lr']})
        return outs.loss
    def configure_scheduler(self):
        return get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps=args.iters * training_args.warmup_ratio * 3, 
                                            num_training_steps=args.iters//args.grad_acc)

class Finetuning(ImplicitProblem):
    def training_step(self, batch):
        #print('finetuning')
        for k in batch:
            batch[k] = batch[k].to(device)
        labels=batch['labels']

        logits = self.module(batch)

        #loss = loss_fct(logits.view(-1), labels.view(-1))+self.reg_loss()
        #print(logits.shape,labels.shape)
        loss = loss_fct(logits, labels)+self.reg_loss()
        if args.wandb:
            wandb.log({"finetuning loss": loss.item()})
            wandb.log({'finetuning lr': self.optimizer.param_groups[0]['lr']})
        return loss
    def configure_scheduler(self):
        return get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps=args.iters * training_args.warmup_ratio, 
                                            num_training_steps=args.iters//args.grad_acc)
    def reg_loss(self):
        loss = 0
        count=0
        for (n1, p1), (n2, p2) in zip(
            self.module.backbone.deberta.embeddings.named_parameters(), self.pretrain.module.deberta.embeddings.named_parameters()
        ):
            loss = loss + args.lam * (p1 - p2).pow(2).sum()
            count += p1.numel()
        for (n1, p1), (n2, p2) in zip(
            self.module.backbone.deberta.encoder.named_parameters(), self.pretrain.module.deberta.encoder.named_parameters()
        ):
            loss = loss + args.lam * (p1 - p2).abs().sum()
            count += p1.numel()
        return loss/count
    
class Reweighting(ImplicitProblem):
    def training_step(self, batch):
        #print('reweighting')
        for k in batch:
            batch[k] = batch[k].to(device)
        labels=batch['labels']

        logits = self.finetune.module(batch)
        
        #loss = loss_fct(logits.view(-1), labels.view(-1))+self.reg_loss()
        loss = loss_fct(logits, labels)+self.reg_loss()
        if args.wandb:
            wandb.log({"reweighting loss": loss.item()})
            wandb.log({'reweighting lr': self.optimizer.param_groups[0]['lr']})
            weight=torch.softmax(self.module.weight,0)
            wandb.log({"cl reweight value": weight[0]})
            wandb.log({"mlm reweight value": weight[1]})
            wandb.log({"sop reweight value": weight[2]})
        return loss
    def configure_scheduler(self):
        return get_linear_schedule_with_warmup(self.optimizer, 
                                            num_warmup_steps=args.iters * training_args.warmup_ratio, 
                                            num_training_steps=args.iters//args.grad_acc)
    def reg_loss(self):
        loss = 0
        count=0
        for (n1, p1), (n2, p2) in zip(
            self.finetune.module.backbone.deberta.embeddings.named_parameters(), self.pretrain.module.deberta.embeddings.named_parameters()
        ):
            loss = loss + args.lam * (p1 - p2).pow(2).sum()
            count += p1.numel()
        for (n1, p1), (n2, p2) in zip(
            self.finetune.module.backbone.deberta.encoder.named_parameters(), self.pretrain.module.deberta.encoder.named_parameters()
        ):
            loss = loss + args.lam * (p1 - p2).abs().sum()
            count += p1.numel()
        return loss/count

test_dataloader=get_data_loader_val(args,config)

class LBIEngine(Engine):
    @torch.no_grad()
    def validation(self):
        
        
        torch.save(self.pretrain.module, '{}/model_pretrain.pth'.format(save_dir))
        torch.save(self.finetune.module, '{}/model_finetune.pth'.format(save_dir))
        torch.save(self.reweight.module, '{}/model_reweight.pth'.format(save_dir))
        model_test=deepcopy(self.finetune.module)
        if args.task=='stsb':
            #metric = evaluate.load("spearmanr")
            metric = evaluate.load("pearsonr")
        elif args.task=='cola':
            metric = evaluate.load("matthews_correlation")
        else: 
            metric = evaluate.load("accuracy")
        
        with torch.no_grad():
            for batch in test_dataloader:
                for k in batch:
                    batch[k] = batch[k].to(device)
                if args.task=='stsb':
                    preds=model_test(batch).reshape(-1).detach().cpu().numpy()
                else:
                    preds=model_test(batch).argmax(-1).detach().cpu().numpy()

                refs = batch['labels'].detach().cpu().numpy()
                #print(preds,refs)
                metric.add_batch(references=refs, predictions=preds)
                del batch
                #break
        result=metric.compute()
        weight=torch.softmax(self.reweight.module.weight,0)
        if args.wandb:
            #wandb.log({"cl reweight value": weight[0]})
            #wandb.log({"mlm reweight value": weight[1]})
            #wandb.log({"sop reweight value": weight[2]})
            wandb.log({"results": result})
        print(weight)
        del model_test
        return {'test':'test'}

if args.load_save:
    model_ft = torch.load('{}/model_finetune.pth'.format(save_dir)).to(device)
    model = torch.load('{}/model_pretrain.pth'.format(save_dir)).to(device)
    model_reweight = torch.load('{}/model_reweight.pth'.format(save_dir)).to(device)
#Define Optimizers

optimizer_finetune = torch.optim.Adam(model_ft.parameters(), lr=args.finetune_lr, weight_decay=training_args.weight_decay)
optimizer_pretrain = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=training_args.weight_decay)
optimizer_reweight = torch.optim.Adam(model_reweight.parameters(), lr=args.reweight_lr, weight_decay=training_args.weight_decay)

# Define configs
reweight_config = Config(type="darts", retain_graph=True, gradient_accumulation=args.grad_acc)
finetune_config = Config(type="darts", allow_unused=True,unroll_steps=1, gradient_accumulation=args.grad_acc)
pretrain_config = Config(type="darts", allow_unused=True,unroll_steps=1, gradient_accumulation=args.grad_acc)
engine_config = EngineConfig(valid_step=args.val_freq, train_iters=args.iters, roll_back=False)

if args.half:
    reweight_config = Config(type="darts", retain_graph=True, precision='bf16', gradient_accumulation=args.grad_acc)
    finetune_config = Config(type="darts", allow_unused=True,unroll_steps=1, precision='bf16', gradient_accumulation=args.grad_acc)
    pretrain_config = Config(type="darts", allow_unused=True,unroll_steps=1, gradient_accumulation=args.grad_acc, precision='bf16') 

reweight = Reweighting(name="reweight", module=model_reweight,optimizer=optimizer_reweight,train_data_loader=eval_dataloader,config=reweight_config)
finetune = Finetuning(name="finetune",module=model_ft,optimizer=optimizer_finetune,train_data_loader=finetune_dataloader ,config=finetune_config)
pretrain = Pretraining(name="pretrain",module=model,optimizer=optimizer_pretrain,train_data_loader=pretrain_dataloader, config=pretrain_config)

problems = [reweight, finetune, pretrain]

u2l = {reweight: [pretrain]}
l2u = {pretrain: [finetune, reweight], finetune: [reweight]}
dependencies = {"u2l": u2l, "l2u": l2u}

engine = LBIEngine(config=engine_config, problems=problems, dependencies=dependencies)
engine.run()

