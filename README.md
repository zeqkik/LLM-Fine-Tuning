# 🃏 AI Uno Master: Fine-tuning a Large Language Model to Play UNO

## Table of Contents
- [Introduction](#introduction)
- [The Problem](#the-problem)
- [The Solution: Fine-tuning an LLM](#the-solution-fine-tuning-an-llm)
- [Project Architecture & Technologies](#project-architecture--technologies)
- [Data Generation: The Gemini Expert](#data-generation-the-gemini-expert)
- [Model Training: LoRA Fine-tuning](#model-training-lora-fine-tuning)
- [Results: Before & After Fine-tuning](#results-before--after-fine-tuning)

## Introduction

This project demonstrates the power of **Large Language Model (LLM) fine-tuning** by teaching a general-purpose LLM (Google's Gemma 7B-IT) how to play the popular card game UNO. From not knowing how to apply the rules consistently, the fine-tuned model transforms into an "AI Uno Master" capable of suggesting correct moves based on game rules.

## The Problem

Generalist LLMs, despite their vast knowledge, often struggle with tasks requiring precise application of specific, domain-bound rules. When prompted with an UNO game scenario, a base LLM might:
- Hallucinate non-existent rules or card effects.
- Provide inconsistent or illogical moves.
- Fail to understand the nuances of action cards (+2, +4, Skip, Reverse).
- Generate verbose or irrelevant explanations.

This project tackles the challenge of bridging this gap, transforming a broad knowledge base into specialized expertise.

## The Solution: Fine-tuning an LLM

The core idea is to fine-tune a pre-trained LLM on a custom dataset of UNO game scenarios and their corresponding correct moves/outcomes. This process adapts the LLM's vast general knowledge to the specific logic and rules of UNO.

## Project Architecture & Technologies

- **Base LLM:** Google's [Gemma 7B Instruction-Tuned (7B-IT)](https://huggingface.co/google/gemma-7b-it)
- **Fine-tuning Method:** Parameter-Efficient Fine-Tuning (PEFT) with [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685)
- **Data Generation LLM:** Google's [Gemini 1.5 Flash API](https://ai.google.dev/models/gemini)
- **Frameworks:** [Hugging Face Transformers](https://huggingface.co/docs/transformers/index), [PEFT](https://huggingface.co/docs/peft/index), [TRL (Transformer Reinforcement Learning)](https://huggingface.co/docs/trl/index), [Datasets](https://huggingface.co/docs/datasets/index)
- **Development Environment:** Google Colab (utilizing A100 GPU)
- **Programming Language:** Python

## Data Generation: The Gemini Expert

To teach the LLM the rules of UNO, a high-quality dataset of `(game scenario, correct move)` pairs was essential. Instead of manual labeling, the powerful **Gemini 1.5 Flash API** was leveraged to act as an "UNO Expert" to synthetically generate this dataset.

- **Process:**
    1.  A comprehensive set of standard UNO rules was formalized and included in every prompt to Gemini.
    2.  A Python script programmatically generated diverse UNO game scenarios (player hand, discard pile).
    3.  Each scenario was submitted to the Gemini 1.5 Flash API with the rules, instructing it to provide the concise, correct move/outcome.
    4.  This process yielded **507 high-quality `(prompt, completion)` pairs**, covering various game states, action cards, and edge cases.
- **Example Data Format (simplified):**
    ```
    {
      "prompt": "Discard: Red 7, Hand: [Blue 3, Red 8, Green 5]. What can you play?",
      "completion": "Red 8. Matches color."
    }
    ```
- **Cost Efficiency:** The generation of 507 samples using Gemini 1.5 Flash incurred a minimal cost, estimated to be around **~0.03 USD**.

## Model Training: LoRA Fine-tuning

The Gemma 7B-IT model was fine-tuned using the generated dataset via LoRA. This technique allowed for efficient training on Google Colab's A100 GPU by only updating a small fraction of the model's parameters.

- **Key Configuration:**
    - **Base Model:** Gemma 7B-IT
    - **Quantization:** 4-bit (using `bitsandbytes`) for memory efficiency.
    - **PEFT Method:** LoRA (rank `r=16`, `lora_alpha=16`, `lora_dropout=0.05`)
    - **Trainer:** `SFTTrainer` from `trl` library
    - **Epochs:** 10 (with early stopping considerations)
    - **Batch Size:** 2 (per device) with gradient accumulation of 4 (effective batch size of 8)
    - **Learning Rate:** 1e-4
    - **Precision:** `fp16=True` (optimized for A100 GPU)

## Results: Before & After Fine-tuning

The transformation of Gemma 7B-IT into an "AI Uno Master" is evident in its ability to consistently apply UNO rules.(**the examples have been coded in Portuguese-BR for publication reasons**)

### Before Fine-tuning (Gemma 7B-IT Base Model)

The base model often struggled with core UNO logic, exhibiting:
- Incorrect move suggestions.
- Hallucinations of non-existent rules or cards.
- Failure to understand action card effects.
- Verbose or poorly formatted responses.

- **Prompt:** `You are playing Uno. Discard: Red 7, Hand: [Blue 3, Red 8, Green 5, +2 Yellow]. What card can you play?`
- **Base Model's Response:** `**Response:** The card you can play is the Green 5.`
- **Analysis:** Incorrect. "Green 5" does not match "Red 7". The correct play would be "Red 8".

### After Fine-tuning (AI Uno Master)

The fine-tuned model consistently provides correct and concise answers, demonstrating its acquired expertise in UNO rules.
- **Prompt:** `You are playing Uno. Discard: Red 7, Hand: [Blue 3, Red 8, Green 5, +2 Yellow]. What card can you play?`
- **Fine-tuned Model's Response:** `Red 8.`
- **Analysis:** Correct! Accurately identified the valid move.

... SHOW THE ACCURACCY...

---
