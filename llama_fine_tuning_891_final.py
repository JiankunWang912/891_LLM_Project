import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--prompt_id', type=int, default=1, help='prompt id')
parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='gpu list')
parser.add_argument('--epoch', type=int, default=3, help='num of epoch')

args = parser.parse_args()

selected_gpus = ','.join(map(str, args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus


from random import randrange
from functools import partial
import torch
from datasets import load_dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          Trainer,
                          TrainingArguments,
                          LlamaTokenizer,
                          DataCollatorWithPadding)

import pandas as pd
from datasets import load_dataset
import bitsandbytes as bnb
import evaluate
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


model_name = "decapoda-research/llama-7b-hf"
load_in_4bit = True
bnb_4bit_use_double_quant = True
bnb_4bit_quant_type = "nf4"
bnb_4bit_compute_dtype = torch.bfloat16


lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
bias = "none"
task_type=TaskType.SEQ_CLS

output_dir = f"./891_Project_results/prompt{args.prompt_id}_gpt"
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 2e-4
optim = "paged_adamw_32bit"
warmup_steps = 2
fp16 = True

dataset_name_tr = "./dataset/Balanced_by_gpt_Adv_cancer_train_70.csv"
dataset_name_val = "./dataset/Adv_cancer_val_20.csv"
dataset_name_te = "./dataset/Adv_cancer_test_10.csv"

seed = 33 
max_length = 2048
num_train_epochs = args.epoch

def create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype):
    """
    Configures model quantization method using bitsandbytes to speed up training and inference

    :param load_in_4bit: Load model in 4-bit precision mode
    :param bnb_4bit_use_double_quant: Nested quantization for 4-bit model
    :param bnb_4bit_quant_type: Quantization data type for 4-bit model
    :param bnb_4bit_compute_dtype: Computation data type for 4-bit model
    """

    bnb_config = BitsAndBytesConfig(
        load_in_4bit = load_in_4bit,
        bnb_4bit_use_double_quant = bnb_4bit_use_double_quant,
        bnb_4bit_quant_type = bnb_4bit_quant_type,
        bnb_4bit_compute_dtype = bnb_4bit_compute_dtype,
    )

    return bnb_config

def load_model(model_name, bnb_config):
    """
    Loads model and model tokenizer

    :param model_name: Hugging Face model name
    :param bnb_config: Bitsandbytes configuration
    """
    # Id to label
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}


    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        trust_remote_code=True,
        quantization_config = bnb_config, 
        device_map='auto', 
        #max_memory = max_memory,
        num_labels=2, 
        id2label=id2label, 
        label2id=label2id,
        cache_dir='/localscratch/wangj306/temp_model/'
)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir='/localscratch/wangj306/temp_model/', unk_token="<unk>", bos_token="<s>", eos_token="</s>")


    # Set padding token as EOS token
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

bnb_config = create_bnb_config(load_in_4bit, bnb_4bit_use_double_quant, bnb_4bit_quant_type, bnb_4bit_compute_dtype)
model, tokenizer = load_model(model_name, bnb_config)

dataset = load_dataset("csv", data_files={"train": dataset_name_tr, "validate": dataset_name_val, "test": dataset_name_te}, cache_dir='./dataset/tmp/')
print(dataset)
print(f'Number of samples [train: {len(dataset["train"])}, valid: {len(dataset["validate"])}, test: {len(dataset["test"])}]')
print(f'Column names are: {dataset["train"].column_names}')



def create_prompt_formats(sample, prompt_id):
    """
    Creates a formatted prompt template for a prompt in the instruction dataset

    :param sample: Prompt or sample from the instruction dataset
    """
    PROMPT = {}
    PROMPT[0] = """Please act as a curator. Based on the input discharge summary, classify if the patient has metastatic cancer or not. Please provide a concise response as either 'Yes' or 'No'."""
    PROMPT[1] = """Please act as a curator. Based on the input discharge summary, classify if the patient has metastatic cancer or not using the following steps.
 Step 1: identify if this patient has cancer or not.
 Step 2: if this patient has cancer, identify its staging, grade, and primary site.
 Step 3: if this patient has metastatic cancer, identify its primary site, and metastatic. 
 Give a final decision 'Yes' or 'No' only."""
    prompt = PROMPT[prompt_id]

    if sample['TEXT'] is None:
        sample['TEXT'] = ''

    if len(prompt) != 0:
        prompt += ' '

    sample['text'] = prompt + sample['TEXT']
    sample['label'] = sample['LABEL']
    return sample


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes dataset batch

    :param batch: Dataset batch
    :param tokenizer: Model tokenizer
    :param max_length: Maximum number of tokens to emit from the tokenizer
    """

    return tokenizer(
        batch["text"],
        max_length = max_length,
        truncation = True,
    )

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed, dataset, prompt_id):
    """
    Tokenizes dataset for fine-tuning

    :param tokenizer (AutoTokenizer): Model tokenizer
    :param max_length (int): Maximum number of tokens to emit from the tokenizer
    :param seed: Random seed for reproducibility
    :param dataset: Instruction dataset (DatasetDict)
    """

    def process_subset(subset, is_test=False):
        # Add prompt to each sample
        _create_prompt_formats = partial(create_prompt_formats, prompt_id = prompt_id)
        subset = subset.map(_create_prompt_formats)

        # Apply preprocessing to each batch of the dataset & remove "TEXT" and "LABEL" fields
        _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
        subset = subset.map(
            _preprocessing_function,
            batched=True,
            remove_columns=["TEXT", "LABEL"],
        )
        
        # Just for logging the information: Filter samples with length less than max_token_length
        filtered_indices = []
        for i, example in enumerate(subset):
            if len(example["input_ids"]) < max_length:
                filtered_indices.append(i)
        all_indices = list(range(len(subset)))
        unfiltered_indices = [i for i in all_indices if i not in filtered_indices]
        with open(f'filtered_indices_{args.prompt_id}.txt', 'w') as file:
            for index in filtered_indices:
                file.write(f"{index}\n")
        with open(f'unfiltered_indices_{args.prompt_id}.txt', 'w') as file:
            for index in unfiltered_indices:
                file.write(f"{index}\n")

        subset = subset.filter(lambda sample: len(sample["input_ids"]) < max_length)

        # shuffle the datasets (training and validation set)
        if not is_test:
            # shuffle non-test datasets
            subset = subset.shuffle(seed=seed)

        return subset

    print("Preprocessing dataset...")

    # Process each subset accordingly
    for subset_key in dataset.keys():
        is_test_subset = subset_key == 'test'
        dataset[subset_key] = process_subset(dataset[subset_key], is_test=is_test_subset)

    return dataset

preprocessed_dataset = preprocess_dataset(tokenizer, max_length, seed, dataset, args.prompt_id)

def create_peft_config(r, lora_alpha, target_modules, lora_dropout, bias, task_type):
    """
    Creates Parameter-Efficient Fine-Tuning configuration for the model

    :param r: LoRA attention dimension
    :param lora_alpha: Alpha parameter for LoRA scaling
    :param modules: Names of the modules to apply LoRA to
    :param lora_dropout: Dropout Probability for LoRA layers
    :param bias: Specifies if the bias parameters should be trained
    """
    config = LoraConfig(
        r = r,
        lora_alpha = lora_alpha,
        target_modules = target_modules,
        lora_dropout = lora_dropout,
        bias = bias,
        task_type = task_type,
    )

    return config

def find_all_linear_names(model):
    """
    Find modules to apply LoRA to.

    :param model: PEFT model
    """

    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    print(f"LoRA module names: {list(lora_module_names)}")
    return list(lora_module_names)

def print_trainable_parameters(model, use_4bit = False):
    """
    Prints the number of trainable parameters in the model.

    :param model: PEFT model
    """

    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    if use_4bit:
        trainable_params /= 2

    print(
        f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
    )

def fine_tune(model,
          tokenizer,
          dataset,
          eval_dataset,
          test_dataset,
          lora_r,
          lora_alpha,
          lora_dropout,
          bias,
          task_type,
          per_device_train_batch_size,
          gradient_accumulation_steps,
          warmup_steps,
          num_train_epochs,
          learning_rate,
          fp16,
          output_dir,
          optim):
    """
    Prepares and fine-tune the pre-trained model.

    :param model: Pre-trained Hugging Face model
    :param tokenizer: Model tokenizer
    :param dataset: Preprocessed training dataset
    """

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    target_modules = find_all_linear_names(model)
    peft_config = create_peft_config(lora_r, lora_alpha, target_modules, lora_dropout, bias, task_type)
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # Training parameters
    trainer = Trainer(
        model = model,
        train_dataset = dataset,
        eval_dataset=eval_dataset,
        args = TrainingArguments(
            output_dir = output_dir, # store the checking points
            logging_dir = f"{output_dir}/log", # store the log
            per_device_train_batch_size = per_device_train_batch_size,
            gradient_accumulation_steps = gradient_accumulation_steps,
            per_device_eval_batch_size= per_device_train_batch_size,
            warmup_steps = warmup_steps,
            num_train_epochs = num_train_epochs,
            learning_rate = learning_rate,
            fp16 = fp16,
            load_best_model_at_end=True,
            optim = optim,
            evaluation_strategy='epoch',
            logging_strategy='epoch',
            save_strategy='epoch',
        ),
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer),
    )

    model.config.use_cache = False

    do_train = True

    print("Training...")

    if do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        # Evaluate on the training dataset after training
        train_predictions = trainer.predict(dataset)
        train_preds = np.argmax(train_predictions.predictions, axis=-1)

        # Compute custom metrics: accuracy, f1, recall, and precision
        accuracy = accuracy_score(train_predictions.label_ids, train_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(train_predictions.label_ids, train_preds, average='binary')

        print(f"Training set performance: Accuracy: {accuracy}, F1: {f1}, Recall: {recall}, Precision: {precision}")
    
    # Evaluation on the validation dataset
    eval_predictions = trainer.predict(eval_dataset)
    eval_preds = np.argmax(eval_predictions.predictions, axis=-1)

    # Compute custom metrics: accuracy, f1, recall, and precision for validation dataset
    eval_accuracy = accuracy_score(eval_predictions.label_ids, eval_preds)
    eval_precision, eval_recall, eval_f1, _ = precision_recall_fscore_support(eval_predictions.label_ids, eval_preds, average='binary')
    print(f"Evaluation set metrics: Accuracy: {eval_accuracy}, F1: {eval_f1}, Recall: {eval_recall}, Precision: {eval_precision}")

    # Evaluation on the testing dataset
    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    
    # Compute custom metrics: accuracy, f1, recall, and precision for testing dataset
    test_accuracy = accuracy_score(predictions.label_ids, preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(predictions.label_ids, preds, average='binary')
    print(f"Testing set metrics: Accuracy: {test_accuracy}, F1: {test_f1}, Recall: {test_recall}, Precision: {test_precision}")

    preds_df = pd.DataFrame(preds)
    preds_df.to_csv(f"./891_Project_results/prompt{args.prompt_id}_gpt/preds.csv")
    true_labels_df = pd.DataFrame(np.array(predictions.label_ids))
    true_labels_df.to_csv(f"./891_Project_results/prompt{args.prompt_id}_gpt/true_labels.csv")

    print("Saving last checkpoint of the model...")
    os.makedirs(output_dir, exist_ok = True)
    trainer.model.save_pretrained(output_dir)

    del model
    del trainer
    torch.cuda.empty_cache()

fine_tune(model,
      tokenizer,
      preprocessed_dataset['train'],
      preprocessed_dataset['validate'],
      preprocessed_dataset['test'],
      lora_r,
      lora_alpha,
      lora_dropout,
      bias,
      task_type,
      per_device_train_batch_size,
      gradient_accumulation_steps,
      warmup_steps,
      num_train_epochs,
      learning_rate,
      fp16,
      output_dir,
      optim)
