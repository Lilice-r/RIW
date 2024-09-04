import argparse


def set_args():

    # Define parser
    parser = argparse.ArgumentParser()

    # Common args for Watermarking and Detecting
    parser.add_argument("--model_name", type=str, default="OPT-1.3B")
    parser.add_argument("--device_id", type=str, default="2")
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--wm_key", type=int, default=34)
    parser.add_argument("--boundary", type=int, default=2)
    # benchmark = [lfqa, opengen, c4]
    parser.add_argument("--benchmark", type=str, default="c4")

    # Watermarking args
    # if benchmark sets to "opengen" or "lfqa", must specify prompt file as the input data file path
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--max_new_tokens", type=int, default=300)
    parser.add_argument("--min_prompt_length", type=int, default=50)
    parser.add_argument("--num_test", type=int, default=500)
    parser.add_argument("--beam_size", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=0.9)

    # Detecting args
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--test_min_tokens", type=int, default=200)
    parser.add_argument("--oracle_model_name", type=str, default="opt-2.7b")

    # Attacking args
    # attack_type = ["sub", "cp", "para", ""]
    parser.add_argument("--attack_type", type=str, default="")
    parser.add_argument("--insertion_ratio", type=int, default=25)
    parser.add_argument("--num_insertions", type=int, default=3)
    parser.add_argument("--replace_rate", type=float, default=0.1)
    parser.add_argument("--chatgpt_temperature", type=float, default=0)

    args = parser.parse_args()
    return args
