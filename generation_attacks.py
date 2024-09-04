import torch
import os
from transformers import AutoTokenizer, LlamaTokenizer
from attack.cp_attack import cp_attack
from attack.sub_attack import sub_attack
from attack.chatgpt_paraphrase_attack import para_attack
from args_config import set_args


if __name__ == "__main__":
    args = set_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    input_file = f"./output/{args.benchmark}/{args.model_name}.jsonl"
    torch.manual_seed(args.wm_key)
    attack_type = args.attack_type

    #  Copy-Paste Attack
    if attack_type == "cp":
        insertion_ratio = args.insertion_ratio
        num_insertions = args.num_insertions
        base_model = args.model_name
        if 'llama' in base_model:
            tokenizer = LlamaTokenizer.from_pretrained(base_model, torch_dtype=torch.float16)
        else:
            tokenizer = AutoTokenizer.from_pretrained(base_model, torch_dtype=torch.float16)
        cp_attack(f_name=input_file,
                  tokenizer=tokenizer,
                  insertion_ratio=insertion_ratio,
                  num_insertions=num_insertions)


    # Substituition Attack
    if attack_type == "sub":
        replace_rate = args.replace_rate
        sub_attack(replace_rate=replace_rate,
                   f_name=input_file)


    # ChatGPT Paraphrase Attack
    if attack_type == "para":
        temperature = args.chatgpt_temperature
        para_attack(f_name=input_file,
                    temperature=temperature)