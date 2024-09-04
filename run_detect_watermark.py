import argparse
import json
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, LlamaTokenizer
from gptwm import GPTWatermarkDetector, load_prior_prob
from args_config import set_args
import os

def main(args):
    input_file = f"./output/{args.benchmark}/{args.model_name}.jsonl"
    output_file = input_file.replace(".jsonl", "_gen_results.jsonl")

    with open(input_file, 'r') as f:
        data = [json.loads(x) for x in f.read().strip().split("\n")]
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)

    vocab_size = 50272 if "OPT" in args.model_name else tokenizer.vocab_size

    model_name = args.model_name.split("/")[-1]
    prior_prob = load_prior_prob(model_name)
    detector = GPTWatermarkDetector(fraction=args.fraction,
                                    strength=args.strength,
                                    vocab_size=vocab_size,
                                    watermark_key=args.wm_key,
                                    boundary=args.boundary,
                                    prior_prob=prior_prob,)

    z_score_list = []
    pred_sum = 0
    for idx, cur_data in tqdm(enumerate(data), total=len(data)):
        if args.attack_type == "para":
            load_data = cur_data["chatgpt_attack_completion"]
            test_min_tokens = 0
            prefix = args.attack_type
        elif args.attack_type == "sub":
            load_data = cur_data["sub_attacked_text"]
            test_min_tokens = 0
            prefix = args.attack_type
        elif args.attack_type == "cp":
            load_data = cur_data["cp_attack_completion"]
            test_min_tokens = 0
            prefix = args.attack_type
        else:
            load_data = cur_data["gen_completion"][0]
            test_min_tokens = args.test_min_tokens
            prefix = "gen"

        gen_tokens = tokenizer(load_data, add_special_tokens=False)["input_ids"]
        if len(gen_tokens) >= test_min_tokens:
            if args.attack_type == "cp":
                pz_score_list = []
                pp_value_list = []
                nz_score_list = []
                np_value_list = []
                window_size = 100
                for i in range(len(gen_tokens) - window_size):
                    seg_pz_score, seg_pp_value, seg_nz_score, seg_np_value = detector.double_detect(gen_tokens[i: i + window_size])
                    pz_score_list.append(seg_pz_score)
                    pp_value_list.append(seg_pp_value)
                    nz_score_list.append(seg_nz_score)
                    np_value_list.append(seg_np_value)
                pz_score = max(pz_score_list)
                pp_value = 0
                nz_score = min(nz_score_list)
                np_value = 0
            else:
                pz_score, pp_value, nz_score, np_value = detector.double_detect(gen_tokens)

            pred = 1 if pz_score > args.threshold or nz_score < -args.threshold else 0
            pred_sum += pred
            cur_data.update({f"{prefix}_pz_score": pz_score, f"{prefix}_pp_value": pp_value, f"{prefix}_nz_score": nz_score, f"{prefix}_np_value": np_value, f"{prefix}_pred": pred})
            z_score_list.append(json.dumps(cur_data))
        else:
            print(f"Warning: sequence {idx} is too short to test.")


    print(z_score_list[:4])
    print("--Valid Number of Samples: ", len(z_score_list))
    print("--Accuracy: ", pred_sum / len(z_score_list))
    with open(output_file, 'w') as f:
        f.write("\n".join(z_score_list) + "\n")


if __name__ == "__main__":
    args = set_args()
    print(args)
    main(args)