from transformers import AutoTokenizer, LlamaTokenizer
import torch
import tqdm
from datasets import load_dataset, Dataset
from args_config import set_args

def cal_word_freq(tokenizer, dataset, normalized=True, save=True, output_file_path=""):
    vocab = list(tokenizer.get_vocab().values())
    vocab_size = len(vocab)
    total_token_freq = torch.zeros([vocab_size])

    for sample in tqdm.tqdm(dataset):
        text = sample["text"]
        sample_idx = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        # print(sample_idx)
        token_freq = torch.zeros((sample_idx.size()[0], vocab_size))
        idx = sample_idx.unsqueeze(0).transpose(0, 1)
        token_freq.scatter_(1, idx, 1)
        token_freq = token_freq.sum(dim=0)
        # print(token_freq)
        # print("--------------")
        total_token_freq = total_token_freq + token_freq
        # print(total_token_freq)
    if normalized:
        total_token_freq = total_token_freq.float() / total_token_freq.sum()
    if save:
        torch.save(total_token_freq, output_file_path)
    return total_token_freq



if __name__ == "__main__":
    args = set_args()
    if 'llama' in args.model_name:
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, torch_dtype=torch.float16)
    dataset = load_dataset("c4", split="train")

    ############################################################################################
    # Calculate the watermark base frequency, calculate three frequencies and average them
    ############################################################################################
    seed = 123
    dataset = dataset.shuffle(seed)
    token_freq_1_dataset = dataset.select(range(0, 100000))
    token_freq_2_dataset = dataset.select(range(100000, 200000))
    token_freq_3_dataset = dataset.select(range(200000, 300000))

    cal_word_freq(tokenizer=tokenizer,
                  dataset=token_freq_1_dataset,
                  normalized=False,
                  save=True,
                  output_file_path=f"token_freq/{args.model_name}/token_freq_1.pt")

    cal_word_freq(tokenizer=tokenizer,
                  dataset=token_freq_2_dataset,
                  normalized=False,
                  save=True,
                  output_file_path=f"token_freq/{args.model_name}/token_freq_2.pt")

    cal_word_freq(tokenizer=tokenizer,
                  dataset=token_freq_3_dataset,
                  normalized=False,
                  save=True,
                  output_file_path=f"token_freq/{args.model_name}/token_freq_3.pt")