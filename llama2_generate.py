import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', nargs='+', type=int, default=[0], help='gpu list')
parser.add_argument('--model_id', type=str, default='7b-chat', choices=['7b', '7b-chat', '13b', '13b-chat', '70b', '70b-chat'], help='llama2 model id')
parser.add_argument('--index', type=int, default=1, help='run times')
parser.add_argument('--temperature', type=float, default=0.9, help='temperature')

args = parser.parse_args()
selected_gpus = ','.join(map(str, args.gpu))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = selected_gpus
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"

model_id = f'meta-llama/Llama-2-{args.model_id}-hf'

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

import pandas as pd
import transformers
from datasets import load_dataset

devset_path = f"Adv_cancer_train_70_positive_part_new.csv"

dataset = load_dataset("csv", data_files={"dev": devset_path}, cache_dir='./load_tmp/')

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# begin initializing HF items, need auth token for these
hf_auth = ''
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

# initialize the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)
model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,
    do_sample=True,
    temperature=args.temperature,
    task='text-generation',
    max_length = 4096
)

def generate_new_summary(record):
    prompt = "This is a discharge summary for a patient. Please act as a healthcare professional and help me rephrase this medical record."
    combined_text = prompt + "\n" + record

    # Tokenize the combined text and check the length
    tokens = tokenizer.encode(combined_text, add_special_tokens=False, truncation=True, max_length=2048)
    if len(tokens) > 2048:
        tokens = tokens[:2048]
    combined_text = tokenizer.decode(tokens, clean_up_tokenization_spaces=True)
    print(combined_text)

    # generate the response
    new_summary = generate_text(combined_text + "\n" )[0]['generated_text']
    out_put_summary = new_summary.lstrip(''.join(filter(lambda x: not x.isalnum(), set(new_summary))))
    print(out_put_summary)
    return out_put_summary

dataset = dataset.map(lambda example: {'TEXT': generate_new_summary(example['TEXT'])})

# Convert HuggingFace datasets to pandas DataFrames
dev_df = dataset["dev"].to_pandas()

# Define the file paths for the new CSV files
new_devset_path = f"Adv_cancer_train_70_positive_augmented_by_llama2_{args.model_id}_{args.index}_part.csv"

# Write the DataFrames to new CSV files
dev_df.to_csv(new_devset_path, index=False)

print("New Test CSV files have been saved.")