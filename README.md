Subtle Signatures, Strong Shields: Advancing Robust and Imperceptible Watermarking in Large Language Models
=========================

This repo contains the *PyTorch* code for paper [Subtle Signatures, Strong Shields: Advancing Robust and Imperceptible Watermarking in Large Language Models].


## Requirements

- transformers==4.15.0
- torch==1.10.1

## LLMs
GPT2-XL, OPT-1.3B and LLaMA2-7B 

## Dataset

C4, OpenGen and LFQA

## Running
Natural frequency statistic of C4 with:
```
python token_freq_cal.py
```

Generate watermarking texts with:
```
python run_generate.py
```

Detect watermark with:
```
python run_detect_watermark.py
python run_detect_unmark.py
```

Text Quality Evaluation with:
```
python run_fluency.py
```






