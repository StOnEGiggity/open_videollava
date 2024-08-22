# (Ongoing Project) LLMs within continual learning for OpenEQA

This repo is based on the repo OpenEQA: Embodied Question Answering in the Era of Foundation Models

## Abstract

To relsove the catastrophic forgetting challenge for LLMs, we plan to leverage continual learning to learn new knowledge continually because the cost of training for LLMs is expensive. Here we refactor the evaluation code for different LLMs on OpenEQA dataset. Then, we will provide training code soon.

## Dataset

The OpenEQA dataset consists of 1600+ question answer pairs $(Q,A^*)$ and corresponding episode histories $H$.

The question-answer pairs are available in [data/open-eqa-v0.json](data/open-eqa-v0.json) and the episode histories can be downloaded by following the instructions [here](data).

**Preview:** A simple tool to view samples in the dataset is provided [here](viewer).

## Baselines and Automatic Evaluation

### Installation

The code requires a `python>=3.9` environment. We recommend using conda:

```bash
conda create -n openeqa python=3.9
conda activate openeqa
pip install -r requirements.txt
pip install -e .
```

### Running baselines

Several baselines are implemented in [openeqa/baselines](openeqa/baselines). In general, baselines are run as follows:

```bash
# set an environment variable to your personal API key for the baseline
python openeqa/baselines/<baseline>.py --dry-run  # remove --dry-run to process the full benchmark
```

See [openeqa/baselines/README.md](openeqa/baselines/README.md) for more details.

### Running evaluations

Automatic evaluation is implemented with GPT-4 using the prompts found [here](prompts/mmbench.txt) and [here](prompts/mmbench-extra.txt).

```bash
# set the OPENAI_API_KEY environment variable to your personal API key
python evaluate-predictions.py <path/to/results/file.json> --dry-run  # remove --dry-run to evaluate on the full benchmark
```

## Reference

```tex
@inproceedings{majumdar2023openeqa,
  author={Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, Karmesh Yadav, Qiyang Li, Ben Newman, Mohit Sharma, Vincent Berges, Shiqi Zhang, Pulkit Agrawal, Yonatan Bisk, Dhruv Batra, Mrinal Kalakrishnan, Franziska Meier, Chris Paxton, Sasha Sax, Aravind Rajeswaran},
  title={{OpenEQA: Embodied Question Answering in the Era of Foundation Models}},
  booktitle={{CVPR}},
  year={2024},
}
```
