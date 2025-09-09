---
title: High level tour of Neural Networks
---

:::{image} ./tip_of_the_iceberg_assets/tip-of-the-iceberg.svg
width: 500px
:::

Before we dive deep into the depths of the neural networks iceberg, I want to
take you through a tour of the tip of the iceberg. While we are on this
tour, we will set up some terminology, and understand some terms that people
use while talking about neural networks.

# Teachable machines

Our tour starts with a very simple demo of [teachablemachines.google.com](https://teachablemachine.withgoogle.com/train).
On this tool no programming is required.
Still this is an excellent tool to understand the following terminology and
concepts

## Glossary of terms to understand through this exercises


:::{glossary}
Training
: The process of finding weights that fit the data.

Training data
: Data with labels that you train the model on.

Classes
: The categories that labels can take.

Model weights/checkpoint
: The weights corresponding to the model.

Inference
: Using the model to predict the label.

Test data
: Sometimes a part of collected data is kept aside to evaluate or
test the model.
:::


# Gemma 3 on Huggingface

Create an account on [Hugging face](https://huggingface.co/google/gemma-3-12b-it) and try it.

# Gemma 3 through Hugging face

Websites can hide a lot of complexity. Any open source website can be run
locally. That takes away a lot of magic away, but still a lot details remain.
Let's try to run a model on `bridges2.psc.edu`.

## Login

```bash
ssh vdhiman@bridges2.psc.edu
```
## Get node allocation

```bash
interact --nodes 1 --ntasks-per-node=4 --mem=22000 --time 2:00:00 -gpu
```

## Load PyTorch singularity image

```bash
singularity shell /ocean/containers/ngc/pytorch/pytorch_24.08-py3.sif
```

```bash
export HF_HOME=/ocean/projects/cis250148p/vdhiman/.cache/huggingface/
pip install huggingface_hub transformers
```

## Create a Python file

:::{code} python
filename: gemma3.py
import torch
from transformers import pipeline

pipe = pipeline(
    "image-text-to-text",
    model="google/gemma-3-12b-it", # "google/gemma-3-12b-it", "google/gemma-3-27b-it" 
    device="cuda",
    torch_dtype=torch.bfloat16
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
:::




# References

https://apxml.com/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/rule-thumb-vram

https://apxml.com/courses/llm-model-sizes-hardware/chapter-5-estimating-hardware-needs/activation-memory

https://github.com/huggingface/blog/blob/main/gemma3.md

https://huggingface.co/collections/google/siglip-659d5e62f0ae1a57ae0e83ba

https://arxiv.org/pdf/2303.15343

https://huggingface.co/spaces/huggingface-projects/gemma-3-12b-it

https://huggingface.co/docs/transformers/v4.56.1/en/model_doc/gemma3#notes

https://github.com/google/gemma_pytorch?tab=readme-ov-file

https://ai.google.dev/gemma/docs/core/pytorch_gemma

# Additional notes

Cant run https://colab.research.google.com/drive/1he3iSSaNzqI7kj8rCO2wfaNZYm0IT8PB?usp=sharing
because of a silly enumeration mismatch?

Dependency hell on bridges2.psc.edu Pytorch  does not suport V100  with
compute 7. Huggingface needs torch.distributed.tensor. Works with singularity
containter  /ocean/containers/ngc/pytorch/pytorch_24.08-py3.sif

Inference time
1.7 * Params * Param size

Nano GPT

https://github.com/karpathy/nanoGPT

MLP

https://colab.research.google.com/github/wecacuee/ECE490-Neural-Networks/blob/master/notebooks/06-pytorch/MLP%20Using%20Pytorch.ipynb#scrollTo=2958611b

Gemma 3 through pytorch is getting killed
