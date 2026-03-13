---
base_model: mufeedh28/dictalm2-israeli-law-instruct-merged
tags:
- gguf
- mistral
- legal
- hebrew
- israel
- law
- ollama
license: apache-2.0
language:
- he
pipeline_tag: text-generation
---

<div align="center">

# DictaLM 2.0 — Israeli Law Chat (GGUF)

### Run the Hebrew legal chatbot locally with Ollama

**F16 full precision** | **~14.5 GB** | **Requires 16GB+ RAM**

</div>

---

## Quick Start

```bash
ollama run hf.co/mufeedh28/dictalm2-israeli-law-GGUF
```

Then ask questions in Hebrew:

```
>>> מהן זכויות השוכר לפי חוק השכירות?
>>> מה קורה אם מעסיק לא משלם פיצויי פיטורים?
>>> האם ניתן לערער על החלטת בית משפט השלום?
```

## About

This is the **F16 (full precision) GGUF** version of [DictaLM 2.0 — Israeli Law Chat](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged), a 7B Hebrew legal chatbot fine-tuned on 140K+ Israeli legal documents and 7,291 Q&A pairs.

For full model details, training data, and usage examples, see the [main model card](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged).

## File Details

| File | Precision | Size | Quality |
|------|:---------:|:----:|:-------:|
| `dictalm2-israeli-law.F16.gguf` | F16 | ~14.5 GB | Full precision — no quality loss |

## Requirements

- [Ollama](https://ollama.com/) installed
- 16 GB+ RAM (GPU or CPU)

## Alternative Usage

### With llama.cpp directly

```bash
./llama-cli -m dictalm2-israeli-law.F16.gguf -p "[INST] מהן זכויות העובד בפיטורים? [/INST]" -n 512
```

### With llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(model_path="dictalm2-israeli-law.F16.gguf", n_ctx=2048)
output = llm("[INST] מהן זכויות העובד בפיטורים? [/INST]", max_tokens=512, temperature=0.7)
print(output["choices"][0]["text"])
```

---

> **Disclaimer:** This model may produce inaccurate legal information. Do not use as a substitute for professional legal advice.

Made by [Mufeed Hammud](https://www.linkedin.com/in/mufeed-hammud-a41b84245) | [Full Model](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged) | [GitHub](https://github.com/mofeed28/israeli-law-llm)
