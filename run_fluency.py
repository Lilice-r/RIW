import argparse
import json
from tqdm import tqdm
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
# from gptwm import WatermarkDetector
from args_config import set_args
import os

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    input_file = f"./output/{args.benchmark}/{args.model_name}.jsonl"
    output_file = input_file.replace(".jsonl", "_ppl_results.jsonl")
    with open(input_file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    print(args.oracle_model_name)
    if 'llama' in args.oracle_model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.oracle_model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.oracle_model_name, torch_dtype=torch.float16)

    model = AutoModelForCausalLM.from_pretrained(args.oracle_model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    def calculate_ppl(prefix, gen_tokens, model):
        gen_inputs = torch.cat([prefix, gen_tokens], dim=-1).to(model.device)
        gen_labels = gen_inputs.clone().detach()
        gen_labels[:, : prefix.size(1)] = -100
        with torch.inference_mode():
            outputs = model(input_ids=gen_inputs, labels=gen_labels)
        loss = outputs.loss
        ppl = torch.tensor(math.exp(loss))

        return ppl.item()

    result_list = []
    sum_gen_ppl = 0
    sum_gold_ppl = 0
    sum_z_score = 0
    for idx, cur_data in tqdm(enumerate(data), total=len(data)):
        gen_tokens = tokenizer(cur_data["gen_completion"][0], add_special_tokens=False, return_tensors="pt")["input_ids"]
        prefix = tokenizer(cur_data["prefix"], add_special_tokens=False, return_tensors="pt")["input_ids"]
        gold_tokens = tokenizer(cur_data["gold_completion"], add_special_tokens=False, return_tensors="pt")["input_ids"]

        if gen_tokens.size(1) >= args.test_min_tokens:
            sum_z_score += max([-cur_data["gen_nz_score"], cur_data["gen_pz_score"]])
            gen_ppl = calculate_ppl(prefix=prefix, gen_tokens=gen_tokens, model=model)
            gold_ppl = calculate_ppl(prefix=prefix, gen_tokens=gold_tokens, model=model)
            sum_gen_ppl += gen_ppl
            sum_gold_ppl += gold_ppl
            cur_data.update({"wm_ppl": gen_ppl, "gold_ppl": gold_ppl})
            result_list.append(json.dumps(cur_data))
        else:
            print(f"Warning: sequence {idx} is too short to test.")


    print(result_list[: 4])
    print("--Valid Number of Samples: ", len(result_list))
    print("--Averaged Generation z-score: ", sum_z_score / len(result_list))
    print("--Averaged Generation PPL: ", sum_gen_ppl / len(result_list))
    print("--Averaged Gold PPL: ", sum_gold_ppl / len(result_list))
    with open(output_file, 'w') as f:
        f.write("\n".join(result_list) + "\n")

    print('Finished!')


if __name__ == "__main__":
    args = set_args()
    print(args)
    main(args)