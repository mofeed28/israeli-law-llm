---
base_model: dicta-il/dictalm2.0
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
license: apache-2.0
language:
- he
pipeline_tag: text-generation
datasets:
- mufeedh28/israeli-law-pretrain
model-index:
- name: DictaLM 2.0 - Israeli Law
  results: []
---

# DictaLM 2.0 - Israeli Law

A Hebrew legal language model fine-tuned on 140,000+ Israeli legal documents. Built for understanding, generating, and working with Israeli law, court rulings, and civil rights content.

## Model Details

| | |
|---|---|
| **Base Model** | [dicta-il/dictalm2.0](https://huggingface.co/dicta-il/dictalm2.0) (Mistral-based, 7B params) |
| **Training** | Continued pretraining with QLoRA (4-bit) via [Unsloth](https://github.com/unslothai/unsloth) |
| **Language** | Hebrew |
| **Domain** | Israeli law, court rulings, legislation, civil rights |
| **License** | Apache 2.0 |

## Training Data

The model was trained on ~140,000 Israeli legal documents from four sources:

| Source | Documents | Description |
|--------|-----------|-------------|
| Israeli Courts (court.gov.il) | ~97,000 | Supreme Court and district court rulings |
| Kol-Zchut (kolzchut.org.il) | ~5,300 | Citizens' rights guides and legal explainers |
| Wikisource Laws | ~3,800 | Israeli legislation and basic laws |
| **Total (after filtering)** | **~106,000** | |

**Data Pipeline:**
- Text cleaning and normalization (niqqud removal, whitespace, template stripping)
- PII scrubbing (Israeli ID numbers, phone numbers, emails, credit cards)
- Quality filtering (minimum length, Hebrew ratio, repetition, boilerplate)
- Near-deduplication via MinHash LSH (threshold 0.7)
- Source balancing: Kol-Zchut and Wikisource upsampled 5x to balance court dominance

## Training Details

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA A100-SXM4-40GB |
| Precision | BF16 + 4-bit QLoRA |
| LoRA rank | 64 |
| LoRA target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 167M / 7.4B (2.26%) |
| Batch size | 16 (4 x 4 gradient accumulation) |
| Learning rate | 2e-4 (cosine schedule) |
| Epochs | 1 |
| Context length | 2,048 tokens |
| Packing | Enabled |
| Training time | ~7.75 hours |
| Framework | Unsloth + HuggingFace TRL |

### Training Loss

| Step | Train Loss | Val Loss |
|------|-----------|----------|
| 500 | 0.850 | 0.827 |
| 1000 | 0.781 | 0.816 |
| 2000 | 0.794 | 0.801 |
| 4000 | 0.697 | 0.782 |
| 6000 | 0.636 | 0.770 |
| 8000 | 0.564 | 0.769 |
| **8785** | **0.700** | **0.769** |

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mufeedh28/dictalm2-israeli-law-merged")
tokenizer = AutoTokenizer.from_pretrained("mufeedh28/dictalm2-israeli-law-merged")

prompt = "בית המשפט העליון פסק כי"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7, top_p=0.9, repetition_penalty=1.15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### With Unsloth (faster inference)

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("mufeedh28/dictalm2-israeli-law-merged")
FastLanguageModel.for_inference(model)

inputs = tokenizer("חוק יסוד: כבוד האדם וחירותו קובע כי", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Outputs

**Prompt:** `חוק יסוד: כבוד האדם וחירותו קובע כי`

> כל אדם זכאי להגנה על חייו, גופו וכבודו. הזכות להגנה עצמית היא חלק מהזכות לכבוד. בנסיבות של הגנה עצמית מותר לאדם לתקוף את מי שמאיים עליו או תוקף אותו כדי להפסיק את התקיפה או למנוע אותה. שימוש בכוח במצב של הגנה עצמית מוגבל למידה הסבירה הדרושה בנסיבות המקרה.

**Prompt:** `על פי חוק השכירות והשאילה`

> שטר-חוב הוא סוג מסוים של ערובה שמותר לדרוש משוכר דירה להפקיד בידי המשכיר. שטר-החוב צריך לעמוד בכל התנאים הבאים כדי שיהיה תקף: שוכרי הדירה יהיו רשומים בו כחייבים, על שטר החוב יופיע סכום כספי, השטר יהיה בר פרעון באופן מיידי עם דרישה של בעל הדירה.

## Intended Use

- Legal text completion and generation
- Hebrew legal NLP research
- Legal document understanding and analysis
- Building legal search and retrieval systems
- Educational tools for Israeli law

## Limitations

- This is a **text completion model**, not a chatbot. It continues text, not answers questions.
- May generate plausible-sounding but incorrect legal information. **Do not use as legal advice.**
- Trained primarily on court rulings — may be less knowledgeable about specific regulatory domains.
- May occasionally reproduce patterns from training data.
- Hebrew-only. Performance on other languages will match the base DictaLM 2.0 model.

## Citation

```bibtex
@misc{dictalm2-israeli-law,
  title={DictaLM 2.0 - Israeli Law},
  author={Mufeed Hammud},
  year={2026},
  url={https://huggingface.co/mufeedh28/dictalm2-israeli-law-merged},
  note={Fine-tuned from dicta-il/dictalm2.0 on Israeli legal corpus}
}
```

## Acknowledgments

- [Dicta - The Israel Center for Text Analysis](https://dicta.org.il/) for the base DictaLM 2.0 model
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- Israeli Courts, Kol-Zchut, and Hebrew Wikisource for the source data
