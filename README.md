# REFINE: Reinforced Fast Weights with Next Sequence Prediction

This repository contains the official implementation of **REFINE**, introduced in the paper:

> **Reinforced Fast Weights with Next Sequence Prediction**  
> Anonymous Authors  
> *Under review at ICML 2026*




## 🔍 Overview

Fast weight architectures (e.g., LaCT, DeltaNet) are typically trained with next-token prediction (NTP), which provides only token-level supervision. REFINE addresses this limitation by optimizing Next-Sequence Prediction (NSP) via reinforcement learning. REFINE is **phase-agnostic** and can be applied during *mid-training, post-training, and test-time training*.

<p align="center">
  <img src="assets/teaser.png" width="85%">
</p>



## 🧠 Method Summary

REFINE consists of four main steps:

1. **Entropy-Based Token Selection**  
   Select informative positions based on NTP entropy.

2. **Rollout Generation**  
   Generate multi-token continuations from truncated prefixes.

3. **Reward Assignment**  
   Compute sequence-level rewards using cosine similarity (or exact match).

4. **Optimization with RL**  
   Optimize NSP using GRPO, combined with standard NTP loss.

<p align="center">
  <img src="assets/main_method.png" width="85%">
</p>


## 📊 Results

REFINE consistently improves long-context performance over supervised fine-tuning (SFT):

- **Needle-in-a-Haystack (RULER)**
- **Multi-document QA**
- **LongBench (12 tasks, up to 16K context)**

<p align="center">
  <img src="assets/ruler.png" width="85%">
</p>

See the paper for detailed tables and ablations.


## 🚀 Getting Started

### Installation

Create a conda environment and install the required dependencies:

```bash
# Create conda environment
conda create -n refine python=3.12 -y
conda activate refine

# Install dependencies
pip install -r requirements.txt

# Install the verl package
cd verl
pip install -e .
```

### Mid-Training

1. Download the [Long-Data-Collections](https://huggingface.co/datasets/togethercomputer/Long-Data-Collections) dataset. Filter for train samples with at least 16K tokens. Save the training dataset in parquet. 
2. Fill in the variables in `examples/refine_trainer/demo/run_midtrain_demo.sh`.
3. Run `examples/refine/demo/run_midtrain_demo.sh`


### Post-Training

1. See [RULER](https://github.com/NVIDIA/RULER) for SQuADQA and HotpotQA multi-doc QA dataset generation. The raw training corpus can be found here: [SQuADQA](https://rajpurkar.github.io/SQuAD-explorer/), [HotpotQA](https://hotpotqa.github.io/). Save the train and test files in parquet format. We provide a sample training dataset in `ReFINE/data/ruler`.
2. Fill in the variables in `run_posttrain_demo.sh`.
3. Run `run_posttrain_demo.sh`


### Test-Time Training

1. Download the raw [LongBench](https://huggingface.co/datasets/yanbingzheng/LongBench/tree/main) dataset. Filter for samples with at most 16K tokens and save as parquet file.
2. Fill in the variables in `run_testtimetrain_demo.sh`.
3. Run `run_testtimetrain_demo.sh`
```

---

## 📝 Citation

If you find this work helpful, please cite our paper:

bibtex tbd 

---

## 🙏 Acknowledgments

This project builds upon [verl](https://github.com/volcengine/verl) for distributed RL training infrastructure.

