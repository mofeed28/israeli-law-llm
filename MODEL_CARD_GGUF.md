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
- quantized
license: apache-2.0
language:
- he
pipeline_tag: text-generation
---

<div align="center">

# DictaLM 2.0 — Israeli Law Chat (GGUF)

### Run the Hebrew legal chatbot locally with Ollama

**Q4_K_M quantized** | **~4 GB** | **Runs on any 8GB+ GPU or CPU**

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

This is the **Q4_K_M quantized GGUF** version of [DictaLM 2.0 — Israeli Law Chat](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged), a 7B Hebrew legal chatbot fine-tuned on 140K+ Israeli legal documents and 7,291 Q&A pairs.

For full model details, training data, benchmarks, and usage examples, see the [main model card](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged).

## File Details

| File | Quantization | Size | Quality |
|------|:-----------:|:----:|:-------:|
| `dictalm2-israeli-law-merged.Q4_K_M.gguf` | Q4_K_M | ~4.4 GB | Good balance of speed and quality |

## Requirements

- [Ollama](https://ollama.com/) installed
- 8 GB+ RAM (GPU or CPU)

## Alternative Usage

### With llama.cpp directly

```bash
./llama-cli -m dictalm2-israeli-law-merged.Q4_K_M.gguf -p "[INST] מהן זכויות העובד בפיטורים? [/INST]" -n 512
```

### With llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama(model_path="dictalm2-israeli-law-merged.Q4_K_M.gguf", n_ctx=2048)
output = llm("[INST] מהן זכויות העובד בפיטורים? [/INST]", max_tokens=512, temperature=0.7)
print(output["choices"][0]["text"])
```

---

> **Disclaimer:** This model may produce inaccurate legal information. Do not use as a substitute for professional legal advice.

Made by [Mufeed Hammud](https://www.linkedin.com/in/mufeed-hammud-a41b84245) | [Full Model](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged) | [GitHub](https://github.com/mofeed28/israeli-law-llm)
