import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset,concatenate_datasets
from transformers import default_data_collator,DataCollatorForLanguageModeling,MODEL_FOR_MASKED_LM_MAPPING,PretrainedConfig,AutoTokenizer,TrainingArguments
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead
import numpy as np
from dataclasses import dataclass, field
from math import prod
from torch.nn import CrossEntropyLoss
from transformers.file_utils import cached_property, torch_required, is_torch_tpu_available
from typing import Optional

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
    "agnews": ("text", None),
    "scierc": ("text", None),
    "rct": ("text", None),
}

class LoraModule(nn.Module):
    #assume only lora layer for query and value layer
    def __init__(self,num_layers,lora_r,in_feat,out_feat,device,config):
        super().__init__()
        self.list_A=nn.ModuleList([nn.Linear(in_feat, lora_r, bias=False) for _ in range(num_layers*2)])
        self.list_B=nn.ModuleList([nn.Linear(lora_r, out_feat, bias=False) for _ in range(num_layers*2)])
        self.clf=RobertaClassificationHead(config)
        self.loss_fct=CrossEntropyLoss()
        self.num_labels=config.num_labels
        self.device=device
    def forward(self,batch,roberta):
        labels=batch['labels'].to(self.device)
        input_dict={k: v.to(self.device) for k, v in batch.items() if k != "labels"}

        input_dict['list_A']= self.list_A
        input_dict['list_B']=self.list_B

        outputs = roberta(**input_dict)
        logits = self.clf(outputs[0])
        #print(logits)
        loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        preds=logits.argmax(dim=-1).detach().cpu().numpy()
        return loss, preds

def get_data_loader(args,config):

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if args.task == 'mnli':
        datasets = load_dataset("glue", args.task,split='train')
        datasets_val = concatenate_datasets([load_dataset("glue", args.task,split='validation_matched'),load_dataset("glue", args.task,split='validation_mismatched')])
    elif args.task == 'imdb': 
        datasets = load_dataset("stanfordnlp/imdb",split='train')
        datasets_val = load_dataset("stanfordnlp/imdb",split='test')
    elif args.task == 'agnews': 
        datasets = load_dataset("fancyzhx/ag_news", split='train')
        datasets_val = load_dataset("fancyzhx/ag_news",split='test')
    elif args.task == 'scierc': 
        datasets = load_dataset("hrithikpiyush/scierc", split='train')
        datasets_val = load_dataset("hrithikpiyush/scierc",split='validation')
    elif args.task == 'rct': 
        datasets = load_dataset("armanc/pubmed-rct20k", split='train')
        datasets_val = load_dataset("armanc/pubmed-rct20k",split='validation')
    else:
        datasets = load_dataset("glue", args.task,split='train')
        datasets_val = load_dataset("glue", args.task,split='validation')

    is_regression = args.task == "stsb"
    if not is_regression:
        if args.task == 'scierc' or args.task == 'rct':
            label_list = sorted(set(datasets["label"]))
        else:
            label_list = datasets.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[args.task]

    padding = "max_length"

    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}

    elif args.task is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    if args.task == 'rct':label_to_id = {v: i for i, v in enumerate(label_list)}
    max_seq_length = tokenizer.model_max_length

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if sentence2_key:
        datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
        datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
    else:
        if args.task=='imdb' or args.task=='agnews':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=[sentence1_key],load_from_cache_file=True)
            datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=[sentence1_key],load_from_cache_file=True)
        elif args.task == 'scierc':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['metadata',sentence1_key],load_from_cache_file=True)
            datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=['metadata',sentence1_key],load_from_cache_file=True)
        elif args.task == 'rct':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['abstract_id','sentence_id',sentence1_key],load_from_cache_file=True)
            datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=['abstract_id','sentence_id',sentence1_key],load_from_cache_file=True)
        else:
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)
            datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)

    if args.task =='scierc' or args.task == 'rct':
        train_dataset,eval_dataset=datasets,datasets_val
    else:
        dataset = datasets.train_test_split(test_size=args.test_size,shuffle=True)
        train_dataset,eval_dataset=dataset['train'],dataset['test']
    #train_dataset=datasets
    #eval_dataset=datasets_val
    print('finetune dataset length',len(train_dataset))
    print('reweight dataset length',len(eval_dataset))
    finetune_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                    batch_size=args.batch_size,drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, collate_fn=default_data_collator,
                                    batch_size=args.batch_size,drop_last=True)
        
    return finetune_dataloader,eval_dataloader

def get_data_loader_val(args,config):

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    if args.task == 'mnli':
        datasets = concatenate_datasets([load_dataset("glue", args.task,split='validation_matched'),load_dataset("glue", args.task,split='validation_mismatched')])
    elif args.task == 'imdb': 
        datasets = load_dataset("stanfordnlp/imdb", split='test')
    elif args.task == 'agnews': 
        datasets = load_dataset("fancyzhx/ag_news", split='test')
    elif args.task == 'scierc': 
        datasets = load_dataset("hrithikpiyush/scierc", split='test')
    elif args.task == 'rct': 
        datasets = load_dataset("armanc/pubmed-rct20k", split='test')
    else:
        datasets = load_dataset("glue", args.task,split='validation')

    is_regression = args.task == "stsb"
    if not is_regression:
        if args.task == 'scierc' or args.task == 'rct':
            label_list = sorted(set(datasets["label"]))
        else:
            label_list = datasets.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[args.task]

    padding = "max_length"

    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}

    elif args.task is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        #print(label_to_id)
    if args.task == 'rct':label_to_id = {v: i for i, v in enumerate(label_list)}

    max_seq_length = tokenizer.model_max_length

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if sentence2_key:
        datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
        #datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
    else:
        if args.task=='imdb' or args.task=='agnews':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=[sentence1_key],load_from_cache_file=True)
        elif args.task == 'scierc':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['metadata',sentence1_key],load_from_cache_file=True)
        elif args.task == 'rct':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['abstract_id','sentence_id',sentence1_key],load_from_cache_file=True)
        else:
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)
        #datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)



    print('dataset length',len(datasets))
    eval_dataloader = DataLoader(datasets, shuffle=True, collate_fn=default_data_collator,
                                    batch_size=args.batch_size,drop_last=False)
        
    return eval_dataloader


def get_data_loader_tr(args,config):

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    if args.task == 'imdb': 
        datasets = load_dataset("stanfordnlp/imdb", split='train')
    elif args.task == 'agnews': 
        datasets = load_dataset("fancyzhx/ag_news", split='train')
    elif args.task == 'scierc': 
        datasets = concatenate_datasets([load_dataset("hrithikpiyush/scierc", split='train'), load_dataset("hrithikpiyush/scierc", split='validation')])
    elif args.task == 'rct': 
        datasets = concatenate_datasets([load_dataset("armanc/pubmed-rct20k", split='train'), load_dataset("armanc/pubmed-rct20k", split='validation')])
    else:
        datasets = load_dataset("glue", args.task,split='train')


    is_regression = args.task == "stsb"
    if not is_regression:
        if args.task == 'scierc' or args.task == 'rct':
            label_list = sorted(set(datasets["label"]))
        else:
            label_list = datasets.features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    sentence1_key, sentence2_key = task_to_keys[args.task]

    padding = "max_length"

    label_to_id = None
    if (
        config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}

    elif args.task is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
    if args.task == 'rct':label_to_id = {v: i for i, v in enumerate(label_list)}
    max_seq_length = tokenizer.model_max_length

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]
        return result

    if sentence2_key:
        datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
        #datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key, sentence2_key],load_from_cache_file=True)
    else:
        if args.task=='imdb' or args.task=='agnews':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=[sentence1_key],load_from_cache_file=True)
        elif args.task == 'scierc':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['metadata',sentence1_key],load_from_cache_file=True)
        elif args.task == 'rct':
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=['abstract_id','sentence_id',sentence1_key],load_from_cache_file=True)
        else:
            datasets = datasets.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)
        #datasets_val = datasets_val.map(preprocess_function, batched=True, remove_columns=["idx", sentence1_key],load_from_cache_file=True)



    print('dataset length',len(datasets))
    eval_dataloader = DataLoader(datasets, shuffle=True, collate_fn=default_data_collator,
                                    batch_size=args.batch_size,drop_last=False)
        
    return eval_dataloader


def argument_parser(parser):
    #parser = argparse.ArgumentParser(description="regularize the target by the source")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--source_domain", type=str, default="BookCorpus")
    parser.add_argument("--target_domain", type=str, default="GLUE")
    parser.add_argument("--task", type=str, default="qqp")
    parser.add_argument("--features_lr", type=float, default=1e-4)
    parser.add_argument("--classifier_lr", type=float, default=1e-3)
    parser.add_argument("--pretrain_lr", type=float, default=2e-5)
    parser.add_argument("--finetune_lr", type=float, default=2e-5)
    parser.add_argument("--reweight_lr", type=float, default=2e-3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--lam", type=float, help="lambda", default=1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--train_portion", type=float, default=0.9)
    parser.add_argument("--baseline", action="store_true", default=False)
    parser.add_argument("--sigmoid", action="store_true", default=False)
    parser.add_argument("--eval", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--val_freq", type=int, default=100)
    parser.add_argument("--load_save", action="store_true", default=False)
    parser.add_argument("--same_dataset", action="store_true", default=False)
    parser.add_argument("--half", action="store_true", default=False)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--grad_acc", type=int, default=1)

    return parser


def sum_params(model):
    cnt=0
    total=0
    for p in model.parameters():
        dims = prod(p.size())
        n = p.cpu().data.numpy()
        cnt+=dims
        total+=np.sum(n)
    
    return total/cnt


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    # SimCSE's arguments
    temp: float = field(
        default=0.05,
        metadata={
            "help": "Temperature for softmax."
        }
    )
    pooler_type: str = field(
        default="cls",
        metadata={
            "help": "What kind of pooler to use (cls, cls_before_pooler, avg, avg_top2, avg_first_last)."
        }
    ) 
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used)."
        }
    )
    do_mlm: bool = field(
        default=False,
        metadata={
            "help": "Whether to use MLM auxiliary objective."
        }
    )
    do_sop: bool = field(
        default=False,
        metadata={
            "help": "Whether to use SOP auxiliary objective."
        }
    )
    mlm_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."
        }
    )
    mlp_only_train: bool = field(
        default=False,
        metadata={
            "help": "Use MLP only during training"
        }
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments. 
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # SimCSE's arguments
    train_file: Optional[str] = field(
        default=None, 
        metadata={"help": "The training data file (.txt or .csv)."}
    )
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    mlm_probability: float = field(
        default=0.15, 
        metadata={"help": "Ratio of tokens to mask for MLM (only effective if --do_mlm)"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate 
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."}
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device
