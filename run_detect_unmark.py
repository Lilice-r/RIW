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
    output_file = input_file.replace(".jsonl", "_golden_results.jsonl")
    test_min_tokens = args.test_min_tokens
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
        gen_tokens = tokenizer(cur_data["gold_completion"], add_special_tokens=False)["input_ids"]
        if len(gen_tokens) >= test_min_tokens:
            pz_score, pp_value, nz_score, np_value = detector.double_detect(gen_tokens)
            pred = 1 if pz_score <= args.threshold or nz_score >= -args.threshold else 0
            pred_sum += pred
            cur_data.update({"gold_pz_score": pz_score, "gold_pp_value": pp_value, "gold_nz_score": nz_score, "gold_np_value": np_value, "gold_pred": pred})
            z_score_list.append(json.dumps(cur_data))
        else:
            print(f"Warning: sequence {idx} is too short to test.")

    print(z_score_list[:4])
    print("--Valid Number of Samples: ", len(z_score_list))
    print("--Accuracy: ", pred_sum / len(z_score_list))
    if len(z_score_list) > 200:
        with open(output_file, 'w') as f:
            f.write("\n".join(z_score_list) + "\n")
    else:
        print(f"The number of samples is too short with {len(z_score_list)}, change filter threshould!")



if __name__ == "__main__":
    args = set_args()
    print(args)
    main(args)