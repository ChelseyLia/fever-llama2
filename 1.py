import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from trl import setup_chat_format
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

save_dir = '/home/s1862623/diss/fever/fever-llama2/'

train_filename = save_dir+'datasets/v1_train.csv'
test_filename = save_dir+'datasets/v1_test.csv'

df_train = pd.read_csv(train_filename,
                 names=["label", "claim", "evidence_text"],
                 encoding="utf-8", encoding_errors="replace")

df_test = pd.read_csv(test_filename,
                 names=["label", "claim", "evidence_text"],
                 encoding="utf-8", encoding_errors="replace")

df_train = df_train[(df_train['label'] == 'SUPPORTS') |
                            (df_train['label'] == 'REFUTES') |
                            (df_train['label'] == 'NOT ENOUGH INFO')]

df_train.reset_index(drop=True, inplace=True)

eval_idx = [idx for idx in df_train.index if idx not in list(train.index) + list(test.index)]
X_eval = df_train[df_train.index.isin(eval_idx)]
X_eval = (X_eval
          .groupby('label', group_keys=False)
          .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))

def generate_prompt(data_point):
    return (f"Claim: {data_point['claim']}\n"
            f"Evidence: {data_point['evidence_text']}\n"
            f"Label: {data_point['label']}\n"
            "Based on the evidence, does it SUPPORT, REFUTE, or provide NOT ENOUGH INFO for the claim?")

def generate_test_prompt(data_point):
    return f"Claim: {data_point['claim']}\nEvidence: {data_point['evidence_text']}\nDoes the evidence SUPPORT, REFUTE, or provide NOT ENOUGH INFO for the claim?"

df_train['prompt'] = df_train.apply(generate_prompt, axis=1)
df_eval['prompt'] = df_eval.apply(generate_prompt, axis=1)

df_test['prompt'] = df_test.apply(generate_test_prompt, axis=1)

y_true = df_test.label
eval_data = Dataset.from_pandas(df_eval[['prompt', 'label']])
train_data = Dataset.from_pandas(df_train[['prompt', 'label']])

def evaluate(y_true, y_pred):
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    mapping = {"SUPPORTS": 2, "REFUTES": 0, "NOT ENOUGH INFO": 1}

    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                         if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

# from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Llama-2-7b-hf"
# model_name = "/kaggle/input/llama-2/pytorch/7b-hf/1"

compute_dtype = getattr(torch, "float16")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=compute_dtype,
    quantization_config=bnb_config,
)

model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          trust_remote_code=True,
                                         )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model, tokenizer = setup_chat_format(model, tokenizer)

def predict(test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(test))):
        prompt = test.iloc[i]["prompt"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens = 50,
                        temperature = 0.0,
                        do_sample=False,

                       )
        result = pipe(prompt)
        answer = result[0]['generated_text'].split()
        if "SUPPORTS" in answer:
            y_pred.append("SUPPORTS")
        elif "REFUTES" in answer:
            y_pred.append("REFUTES")
        elif "NOT ENOUGH INFO" in answer:
            y_pred.append("NOT ENOUGH INFO")
        else:
            y_pred.append("NOT ENOUGH INFO")
    return y_pred

##########
print('######### zero shot ###########')
y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    # target_modules="all-linear",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=save_dir+"logs",                        # directory to save and repository id
    num_train_epochs=3,                       # number of training epochs
    per_device_train_batch_size=1,            # batch size per device during training
    gradient_accumulation_steps=8,            # number of steps before performing a backward/update pass
    gradient_checkpointing=True,              # use gradient checkpointing to save memory
    optim="paged_adamw_32bit",
    save_steps=0,
    logging_steps=25,                         # log every 10 steps
    learning_rate=2e-4,                       # learning rate, based on QLoRA paper
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=0.3,                        # max gradient norm based on QLoRA paper
    max_steps=-1,
    warmup_ratio=0.03,                        # warmup ratio based on QLoRA paper
    group_by_length=True,
    lr_scheduler_type="cosine",               # use cosine learning rate scheduler
    report_to="tensorboard",                  # report metrics to tensorboard
    evaluation_strategy="epoch"               # save checkpoint every epoch
)

trainer = SFTTrainer(
    model=model,
    args=training_arguments,
    train_dataset=train_data,
    eval_dataset=eval_data,
    peft_config=peft_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=1024,
    packing=False,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": False,
    }
)

# Train model
trainer.train()

save_model_dir = save_dir+'models/'
finetuned_model = 'llama-2-7b-fact-checking-fever'

trainer.model.save_pretrained(save_model_dir+ finetuned_model)
tokenizer.save_pretrained(save_model_dir+finetuned_model)

from peft import AutoPeftModelForCausalLM

# finetuned_model = save_model_dir
compute_dtype = getattr(torch, "float16")

model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoPeftModelForCausalLM.from_pretrained(
     save_model_dir+finetuned_model,
     torch_dtype=compute_dtype,
     return_dict=False,
     low_cpu_mem_usage=True,
     device_map='auto',
)

merged_model = model.merge_and_unload()


##########
print('######### fine tuned ###########')
y_pred = predict(X_test, merged_model, tokenizer)
evaluate(y_true, y_pred)


