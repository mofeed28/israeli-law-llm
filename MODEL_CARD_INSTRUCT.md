---
base_model:
- dicta-il/dictalm2.0
- mufeedh28/dictalm2-israeli-law-pretrain-merged
tags:
- text-generation-inference
- transformers
- unsloth
- mistral
- legal
- hebrew
- israel
- law
- nlp
- chat
- instruction-tuning
- lora
- qlora
license: apache-2.0
language:
- he
pipeline_tag: text-generation
datasets:
- mufeedh28/israeli-law-pretrain
library_name: transformers
model-index:
- name: DictaLM 2.0 - Israeli Law Chat
  results: []
widget:
- messages:
  - role: user
    content: מהן זכויות השוכר לפי חוק השכירות?
- messages:
  - role: user
    content: מה קורה אם מעסיק לא משלם פיצויי פיטורים?
- messages:
  - role: user
    content: האם ניתן לערער על החלטת בית משפט השלום?
---

<div align="center">

# DictaLM 2.0 — Israeli Law Chat

### The first open-source Hebrew legal chatbot

**140K+ legal documents** | **7,300 Q&A pairs** | **Two-phase fine-tuning** | **Apache 2.0**

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Language](https://img.shields.io/badge/Language-Hebrew-green.svg)]()
[![Base Model](https://img.shields.io/badge/Base-DictaLM_2.0_(7B)-purple.svg)](https://huggingface.co/dicta-il/dictalm2.0)

[Model Hub](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged) · [GGUF for Ollama](https://huggingface.co/mufeedh28/dictalm2-israeli-law-GGUF) · [Phase 1 Model](https://huggingface.co/mufeedh28/dictalm2-israeli-law-pretrain-merged) · [Training Data](https://huggingface.co/datasets/mufeedh28/israeli-law-pretrain) · [GitHub](https://github.com/mofeed28/israeli-law-llm)

</div>

---

## Overview

**DictaLM 2.0 — Israeli Law Chat** is a 7B-parameter Hebrew language model specialized in Israeli law. It can answer questions about Israeli legislation, court rulings, employment law, tenant rights, civil rights, and more — in natural Hebrew.

The model was built in two phases:

1. **Continued Pretraining** — The base [DictaLM 2.0](https://huggingface.co/dicta-il/dictalm2.0) was trained on 140,000+ Israeli legal documents (court rulings, legislation, and citizens' rights guides) to deeply learn the legal domain.
2. **Instruction Tuning** — The pretrained model was then fine-tuned on 7,291 Hebrew legal Q&A pairs to enable conversational question-answering.

> **Disclaimer:** This model is for research and educational purposes. It may produce inaccurate information. **Do not use as a substitute for professional legal advice.**

---

## Quick Start

### Chat with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "mufeedh28/dictalm2-israeli-law-instruct-merged"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set chat template (Mistral format)
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
)

messages = [{"role": "user", "content": "מהן זכויות העובד בפיטורים?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.15,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Chat with Unsloth (2x faster)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "mufeedh28/dictalm2-israeli-law-instruct-merged",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "האם מותר למעסיק לפטר עובדת בהריון?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, repetition_penalty=1.15)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("[/INST]")[-1].strip()
print(answer)
```

### Run Locally with Ollama

```bash
ollama run hf.co/mufeedh28/dictalm2-israeli-law-GGUF
```

Then chat directly in your terminal:
```
>>> מה הדין לגבי פיצויי פיטורים?
```

---

## Training Pipeline

```
┌─────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│   dicta-il/         │     │  Phase 1: Continued      │     │  Phase 2: Instruction    │
│   dictalm2.0        │────▶│  Pretraining             │────▶│  Tuning                  │
│   (Base Model, 7B)  │     │  140K legal docs         │     │  7,291 Q&A pairs         │
└─────────────────────┘     │  Loss: 0.850 → 0.700     │     │  Loss: 1.63 → 0.87       │
                            └──────────────────────────┘     └──────────────────────────┘
                              dictalm2-israeli-law-            dictalm2-israeli-law-
                              pretrain-merged                  instruct-merged ⭐
```

---

## Training Data

### Phase 1 — Legal Corpus (Continued Pretraining)

140,000+ Israeli legal documents from three authoritative sources:

| Source | Documents | Description |
|--------|:---------:|-------------|
| **Israeli Courts** ([court.gov.il](https://www.court.gov.il)) | ~97,000 | Supreme Court, district and magistrate court rulings |
| **Kol-Zchut** ([kolzchut.org.il](https://www.kolzchut.org.il)) | ~5,300 | Citizens' rights guides, legal explainers, entitlements |
| **Hebrew Wikisource** | ~3,800 | Israeli legislation, Basic Laws, Knesset statutes |
| **Total (after filtering & dedup)** | **~106,000** | |

**Data pipeline applied:**
- Unicode normalization, niqqud removal, whitespace cleanup
- PII scrubbing (Israeli ID numbers, phone numbers, emails, credit cards)
- Quality filtering (minimum length, Hebrew ratio, repetition detection, boilerplate removal)
- Near-deduplication via MinHash LSH (threshold 0.7)
- Source balancing: Kol-Zchut and Wikisource upsampled 5x to counter court dominance

### Phase 2 — Q&A Pairs (Instruction Tuning)

7,291 Hebrew question-answer pairs generated from the legal corpus:

| Source | Q&A Pairs | Topics |
|--------|:---------:|--------|
| **Court Rulings** | ~4,500 | Case law, precedents, judicial reasoning |
| **Kol-Zchut** | ~2,000 | Employment rights, tenancy, social security, disability |
| **Wikisource Laws** | ~800 | Statutory interpretation, Basic Laws, regulations |
| **Total** | **7,291** | |

Each Q&A pair follows the format a regular citizen would use — practical questions with clear, source-grounded answers in Hebrew.

**Format:** [ShareGPT](https://huggingface.co/docs/trl/en/sft_trainer#sharegpt-format)
```json
{
  "conversations": [
    {"role": "user", "content": "מהן זכויות השוכר כאשר המשכיר לא מבצע תיקונים בדירה?"},
    {"role": "assistant", "content": "על פי חוק השכירות והשאילה, התשל\"א-1971, כאשר..."}
  ]
}
```

---

## Training Details

### Phase 1 — Continued Pretraining

| Parameter | Value |
|-----------|-------|
| Base model | [dicta-il/dictalm2.0](https://huggingface.co/dicta-il/dictalm2.0) |
| Method | QLoRA (4-bit NormalFloat) |
| LoRA rank / alpha | 64 / 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 167M / 7.4B (2.26%) |
| Batch size | 16 (4 per device × 4 grad accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 1 |
| Context length | 2,048 tokens |
| Packing | Enabled |
| Training steps | 8,785 |
| Training time | ~7.75 hours |
| GPU | NVIDIA A100-SXM4-40GB |
| Optimizer | AdamW 8-bit |
| Framework | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |

<details>
<summary><b>Phase 1 Loss Curve</b></summary>

| Step | Train Loss | Val Loss |
|-----:|:----------:|:--------:|
| 500 | 0.850 | 0.827 |
| 1,000 | 0.781 | 0.816 |
| 2,000 | 0.794 | 0.801 |
| 4,000 | 0.697 | 0.782 |
| 6,000 | 0.636 | 0.770 |
| 8,000 | 0.564 | 0.769 |
| **8,785** | **0.700** | **0.769** |

</details>

### Phase 2 — Instruction Tuning

| Parameter | Value |
|-----------|-------|
| Base model | [mufeedh28/dictalm2-israeli-law-pretrain-merged](https://huggingface.co/mufeedh28/dictalm2-israeli-law-pretrain-merged) |
| Method | QLoRA (4-bit NormalFloat) |
| LoRA rank / alpha | 32 / 16 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | 83.9M / 7.3B (1.14%) |
| Batch size | 16 (4 per device × 4 grad accumulation) |
| Learning rate | 1e-4 (cosine schedule) |
| Epochs | 2 |
| Context length | 2,048 tokens |
| Packing | Disabled (conversations kept intact) |
| Training steps | 912 |
| Training time | ~20 minutes |
| Peak GPU memory | 16.0 GB / 39.5 GB |
| GPU | NVIDIA A100-SXM4-40GB |
| Optimizer | AdamW 8-bit |
| Framework | [Unsloth](https://github.com/unslothai/unsloth) + [TRL](https://github.com/huggingface/trl) |

<details>
<summary><b>Phase 2 Loss Curve</b></summary>

| Step | Train Loss |
|-----:|:----------:|
| 10 | 1.632 |
| 50 | 1.124 |
| 100 | 1.131 |
| 200 | 1.052 |
| 300 | 1.033 |
| 400 | 1.022 |
| 456 | — *(epoch 1 → 2)* |
| 500 | 0.917 |
| 600 | 0.903 |
| 700 | 0.876 |
| 800 | 0.893 |
| 900 | 0.881 |
| **912** | **0.876** |
| | |
| **Average** | **0.989** |

Loss dropped from 1.63 → 0.87 across 2 epochs, with a clear ~0.1 jump between epoch 1 and 2 as the model saw the data for the second time.

</details>

---

## Chat Template

This model uses the **Mistral chat format**:

```
[INST] שאלת המשתמש כאן [/INST]תשובת המודל כאן</s>
```

If your tokenizer doesn't have a chat template set, apply it manually:

```python
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
)
```

---

## Model Family

| Model | Type | Description | Link |
|-------|------|-------------|------|
| **dictalm2-israeli-law-instruct-merged** | Chat (this model) | Full instruction-tuned model — ask legal questions in Hebrew | [Hub](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged) |
| dictalm2-israeli-law-GGUF | GGUF | Quantized (Q4_K_M) for local inference with Ollama | [Hub](https://huggingface.co/mufeedh28/dictalm2-israeli-law-GGUF) |
| dictalm2-israeli-law-pretrain-merged | Base | Phase 1 only — text completion, no chat ability | [Hub](https://huggingface.co/mufeedh28/dictalm2-israeli-law-pretrain-merged) |
| israeli-law-pretrain | Dataset | Full training data (pretrain + instructions) | [Hub](https://huggingface.co/datasets/mufeedh28/israeli-law-pretrain) |

---

## Intended Use

- Answering questions about Israeli law in Hebrew
- Legal research assistance and document analysis
- Hebrew legal NLP research and benchmarking
- Educational tools for understanding Israeli legislation
- Building legal information retrieval systems
- Prototyping legal tech applications

## Limitations and Risks

> **This model is NOT a lawyer.** It is a research tool.

- **Accuracy:** May generate plausible-sounding but factually incorrect legal information. Always verify with official sources or a licensed attorney.
- **Scope:** Trained primarily on court rulings, citizens' rights guides, and legislation available online. Does not cover all areas of Israeli law equally — regulatory, tax, and military law may be underrepresented.
- **Bias:** Training data reflects the documents available in public databases. Court rulings skew toward cases that reached higher courts. Citizens' rights content reflects the Kol-Zchut editorial perspective.
- **Temporal cutoff:** Training data was collected in early 2026. The model is unaware of legislative changes, new court rulings, or policy updates after that date.
- **Language:** Hebrew only. Performance on Arabic, English, or other languages matches the base DictaLM 2.0 model.
- **Hallucination:** Like all language models, it may cite non-existent laws, invent case numbers, or misattribute legal principles. Critical claims should always be verified.
- **Not legal advice:** Using this model's outputs as the sole basis for legal decisions could lead to serious harm. Professional legal counsel is irreplaceable.

## Ethical Considerations

This model was built with the following principles:

- **PII Protection:** All personally identifiable information (ID numbers, phone numbers, addresses) was scrubbed from training data before use.
- **Open Source:** Released under Apache 2.0 to promote transparency and enable community scrutiny of legal AI.
- **Access to Justice:** Designed to help democratize access to legal information in Hebrew, particularly for communities underserved by existing legal resources.

---

## Technical Specifications

- **Architecture:** Mistral 7B (32 layers, 32 attention heads, 4096 hidden dim)
- **Precision:** BF16
- **Format:** Safetensors
- **Context window:** 2,048 tokens
- **Vocabulary:** 32,768 tokens (Mistral tokenizer)

### Hardware Requirements

| Setup | VRAM Required |
|-------|:------------:|
| Full precision (BF16) | ~14 GB |
| 4-bit quantized (QLoRA) | ~5 GB |
| GGUF Q4_K_M (Ollama) | ~4 GB |

---

## Reproduction

Full training code is available on GitHub: [mofeed28/israeli-law-llm](https://github.com/mofeed28/israeli-law-llm)

```bash
git clone https://github.com/mofeed28/israeli-law-llm.git
cd israeli-law-llm

# Phase 1: Continued pretraining (Colab A100 recommended)
# See train_dictalm.ipynb

# Phase 2: Instruction tuning
# See train_instruct.ipynb
```

---

## Citation

```bibtex
@misc{dictalm2-israeli-law-chat,
  title     = {DictaLM 2.0 - Israeli Law Chat: A Hebrew Legal Question-Answering Model},
  author    = {Mufeed Hammud},
  year      = {2026},
  url       = {https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged},
  note      = {Instruction-tuned from dicta-il/dictalm2.0 on 140K+ Israeli legal documents and 7,291 Q&A pairs}
}
```

---

## Acknowledgments

- [**Dicta** — The Israel Center for Text Analysis](https://dicta.org.il/) for the base DictaLM 2.0 model
- [**Unsloth**](https://github.com/unslothai/unsloth) for enabling efficient fine-tuning
- [**Kol-Zchut**](https://www.kolzchut.org.il/) for comprehensive citizens' rights content
- [**Hebrew Wikisource**](https://he.wikisource.org/) for digitized Israeli legislation
- [**Israeli Courts**](https://www.court.gov.il/) for public access to court rulings

---

<div align="center">

**Built in Israel, for Israeli law, in Hebrew.**

Made by [Mufeed Hammud](https://www.linkedin.com/in/mufeed-hammud-a41b84245)

</div>
