import hashlib
import os
from typing import List
import numpy as np
from scipy.stats import norm
import scipy
import torch
from transformers import LogitsWarper


class GPTWatermarkBase:
    """
    Base class for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self,
                 fraction: float = 0.5,
                 strength: float = 2.0,
                 vocab_size: int = 50257,
                 watermark_key: int = 0,
                 boundary=None,
                 prior_prob=None):

        rng = np.random.default_rng(self._hash_fn(watermark_key))
        mask = np.array([True] * int(fraction * vocab_size) + [False] * (vocab_size - int(fraction * vocab_size)))
        rng.shuffle(mask)
        self.green_list_mask = torch.tensor(mask, dtype=torch.float32)
        self.strength = strength
        assert prior_prob.size()[0] == vocab_size

        self.negative = False
        self.gp_mask, self.gn_mask = self.split_green_list(prior=prior_prob, vocab_size=vocab_size, boundary=boundary)
        self.prior_p = (prior_prob * self.gp_mask).sum().item()
        self.prior_n = (prior_prob * self.gn_mask).sum().item()
        # self.fraction = green_list_prior

    def determine_strength(self, prior, strength, fraction):
        prob_score = prior[self.green_list_ids].sum()
        print(f"Given the green list ids, the averaged prior probability for the green list tokens is {prob_score}.")
        if prob_score > fraction:
            print("Applying negative Strength!")
            self.negative = True
            return prob_score, torch.tensor(-strength)
        else:
            print("Applying positive Strength!")
            return prob_score, torch.tensor(strength)
        # return prob_score, torch.tensor(strength)

    def split_green_list(self, prior, vocab_size, boundary):

        positive_mask = prior.gt(boundary/vocab_size)
        negative_mask = prior.le(boundary/vocab_size)
        gp_mask = positive_mask * self.green_list_mask
        gn_mask = negative_mask * self.green_list_mask
        print(f"Partition the green list into positive and negative ones with boundary {boundary}.")
        print(f"Positive green list contains {gp_mask.sum().item()} tokens, about {gp_mask.sum().item() / self.green_list_mask.sum().item()} of all green list, with prior {(gp_mask*prior).sum().item()}")
        print(f"Negative green list contains {gn_mask.sum().item()} tokens, about {gn_mask.sum().item() / self.green_list_mask.sum().item()} of all green list, with prior {(gn_mask*prior).sum().item()}")
        return gp_mask, gn_mask



    @staticmethod
    def _hash_fn(x: int) -> int:
        """solution from https://stackoverflow.com/questions/67219691/python-hash-function-that-returns-32-or-64-bits"""
        x = np.int64(x)
        return int.from_bytes(hashlib.sha256(x).digest()[:4], 'little')


class GPTWatermarkLogitsWarper(GPTWatermarkBase, LogitsWarper):
    """
    LogitsWarper for watermarking distributions with fixed-group green-listed tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.FloatTensor:
        """Add the watermark to the logits and return new logits."""
        watermark_p = self.strength * self.gp_mask
        watermark_n = -self.strength * self.gn_mask
        new_logits = scores + watermark_p.to(scores.device) + watermark_n.to(scores.device)
        return new_logits


class GPTWatermarkDetector(GPTWatermarkBase):
    """
    Class for detecting watermarks in a sequence of tokens.

    Args:
        fraction: The fraction of the distribution to be green-listed.
        strength: The strength of the green-listing. Higher values result in higher logit scores for green-listed tokens.
        vocab_size: The size of the vocabulary.
        watermark_key: The random seed for the green-listing.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _z_score(num_green: int, total: int, fraction: float) -> float:
        """Calculate and return the z-score of the number of green tokens in a sequence."""
        z_score = (num_green - fraction * total) / np.sqrt(fraction * (1 - fraction) * total)
        p_value = scipy.stats.norm.sf(z_score)
        return z_score, p_value

    @staticmethod
    def _compute_tau(m: int, N: int, alpha: float) -> float:
        """
        Compute the threshold tau for the dynamic thresholding.

        Args:
            m: The number of unique tokens in the sequence.
            N: Vocabulary size.
            alpha: The false positive rate to control.
        Returns:
            The threshold tau.
        """
        factor = np.sqrt(1 - (m - 1) / (N - 1))
        tau = factor * norm.ppf(1 - alpha)
        return tau

    def detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""
        green_tokens = 0
        for i in sequence:
            if i in self.green_list_ids:
                green_tokens += 1

        return self._z_score(green_tokens, len(sequence), self.fraction.item())

    def double_detect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value."""

        gp_tokens = int(sum(self.gp_mask[i] for i in sequence))
        gn_tokens = int(sum(self.gn_mask[i] for i in sequence))
        positive_z_score, positive_p_value = self._z_score(gp_tokens, len(sequence), self.prior_p)
        negative_z_score, negative_p_value = self._z_score(gn_tokens, len(sequence), self.prior_n)

        return positive_z_score, positive_p_value, negative_z_score, negative_p_value

    def unidetect(self, sequence: List[int]) -> float:
        """Detect the watermark in a sequence of tokens and return the z value. Just for unique tokens."""
        sequence = list(set(sequence))
        green_tokens = int(sum(self.green_list_mask[i] for i in sequence))
        return self._z_score(green_tokens, len(sequence), self.fraction)

    def dynamic_threshold(self, sequence: List[int], alpha: float, vocab_size: int) -> (bool, float):
        """Dynamic thresholding for watermark detection. True if the sequence is watermarked, False otherwise."""
        z_score = self.unidetect(sequence)
        tau = self._compute_tau(len(list(set(sequence))), vocab_size, alpha)
        return z_score > tau, z_score


def tokenize_and_truncate(tokenizer, example, max_new_tokens, min_prompt_length, model_max_seq_len=512):
    inputs = tokenizer.encode(example["text"], return_tensors="pt", truncation=True, max_length=model_max_seq_len)
    input_length = inputs.shape[1]
    if input_length > max_new_tokens + min_prompt_length:
        prefix_inputs = inputs[:, :input_length - max_new_tokens]
        prefix = tokenizer.batch_decode(prefix_inputs, skip_special_tokens=True)[0]
        gold_completion = tokenizer.batch_decode(inputs[:, input_length - max_new_tokens:], skip_special_tokens=True)[0]
        return {
            "input_ids": prefix_inputs,
            "prefix": prefix,
            "gold_completion": gold_completion,
        }
    else:
        return None


def load_prior_prob(model_name):
    freq_path = f"token_freq/{model_name}"
    token_freq1 = torch.load(os.path.join(freq_path, "token_freq_1.pt"))
    token_freq2 = torch.load(os.path.join(freq_path, "token_freq_2.pt"))
    token_freq3 = torch.load(os.path.join(freq_path, "token_freq_3.pt"))
    avg_token_freq = (token_freq1 / token_freq1.sum() + token_freq2 / token_freq2.sum() + token_freq3 / token_freq3.sum()) / 3
    if "OPT" in model_name:
        avg_token_freq = torch.cat([avg_token_freq, torch.zeros(50272 - avg_token_freq.size()[0])])
    # sort_freq, _ = torch.sort(avg_token_freq, descending=True)
    return avg_token_freq
