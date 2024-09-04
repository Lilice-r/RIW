import argparse
from tqdm import tqdm
import json
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer, LogitsProcessorList
from gptwm import GPTWatermarkLogitsWarper, tokenize_and_truncate, load_prior_prob
from datasets import load_dataset, Dataset
from args_config import set_args

def read_file(filename):
    with open(filename, "r") as f:
        return [json.loads(line) for line in f.read().strip().split("\n")]


def write_file(filename, data):
    with open(filename, "a") as f:
        f.write("\n".join(data) + "\n")


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    output_dir = os.path.join(args.output_dir, args.benchmark)
    if os.path.isdir(output_dir):
        print(f"Output dir already exist. Saving to {output_dir}")
    else:
        os.makedirs(output_dir)
    model_name = args.model_name.split("/")[-1]
    output_file = f"{output_dir}/{model_name.replace('/', '-')}.jsonl"

    if os.path.isfile(output_file):
        print(f"Find previous results, please make sure run in the same setting!")
        raise EnvironmentError

    prior_prob = load_prior_prob(model_name)
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    watermark_processor = LogitsProcessorList([GPTWatermarkLogitsWarper(fraction=args.fraction,
                                                                        strength=args.strength,
                                                                        vocab_size=model.config.vocab_size,
                                                                        watermark_key=args.wm_key,
                                                                        boundary=args.boundary,
                                                                        prior_prob=prior_prob)])

    if args.benchmark == "c4":
        dataset = load_dataset("c4",
                               split="train")
        seed = 123
        dataset = dataset.shuffle(seed)
        dataset = dataset.select(range(300000, 303000))
    else:
        dataset = read_file(args.prompt_file)

    num_cur_outputs = 0

    outputs = []

    generate_args = {
        'logits_processor': watermark_processor,
        'output_scores': True,
        'return_dict_in_generate': True,
        'max_new_tokens': args.max_new_tokens,
    }

    if args.beam_size is not None:
        generate_args['num_beams'] = args.beam_size
    else:
        generate_args['do_sample'] = True
        generate_args['top_k'] = args.top_k
        generate_args['top_p'] = args.top_p

    torch.manual_seed(args.wm_key)
    for idx, cur_data in tqdm(enumerate(dataset), total=len(dataset)):
        if idx < num_cur_outputs or len(outputs) >= args.num_test:
            continue

        # Different Load Data Method
        if args.benchmark == "c4":
            inputs = tokenize_and_truncate(tokenizer=tokenizer,
                                           example=cur_data,
                                           max_new_tokens=args.max_new_tokens,
                                           min_prompt_length=args.min_prompt_length,
                                           model_max_seq_len=model.config.max_position_embeddings,)
            if inputs is None:
                continue
            else:
                prefix = inputs["prefix"]
                gold_completion = inputs["gold_completion"]
                input_ids = inputs["input_ids"]
        else:
            if "gold_completion" not in cur_data and 'targets' not in cur_data:
                continue
            elif "gold_completion" in cur_data:
                prefix = cur_data['prefix']
                gold_completion = cur_data['gold_completion']
            else:
                prefix = cur_data['prefix']
                gold_completion = cur_data['targets'][0]

            batch = tokenizer(prefix, truncation=True, return_tensors="pt")
            input_ids = batch['input_ids']


        # Watermark Generation Method
        num_tokens = len(input_ids[0])
        with torch.inference_mode():

            generation = model.generate(input_ids.to(model.device), **generate_args)
            gen_text = tokenizer.batch_decode(generation['sequences'][:, num_tokens:], skip_special_tokens=True)

        outputs.append(json.dumps({
            "prefix": prefix,
            "gold_completion": gold_completion,
            "gen_completion": gen_text
        }))

        if (idx + 1) % 100 == 0:
            write_file(output_file, outputs)
            outputs = []

    write_file(output_file, outputs)
    print("Finished!")


if __name__ == "__main__":
    args = set_args()
    print(args)
    main(args)