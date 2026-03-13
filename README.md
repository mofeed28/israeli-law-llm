# Israeli Law LLM

The first open-source Hebrew legal chatbot. Fine-tuned on 140,000+ Israeli legal documents and 7,291 Q&A pairs, built on [DictaLM 2.0](https://huggingface.co/dicta-il/dictalm2.0) (7B parameters).

**Chat Model:** [mufeedh28/dictalm2-israeli-law-instruct-merged](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged)
**Run Locally:** `ollama run hf.co/mufeedh28/dictalm2-israeli-law-GGUF`

## Quick Start

### Run locally with Ollama

```bash
ollama run hf.co/mufeedh28/dictalm2-israeli-law-GGUF
```

### Use with Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mufeedh28/dictalm2-israeli-law-instruct-merged", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mufeedh28/dictalm2-israeli-law-instruct-merged")

tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]"
    "{% elif message['role'] == 'assistant' %}{{ message['content'] }}{{ eos_token }}"
    "{% endif %}{% endfor %}"
)

messages = [{"role": "user", "content": "מהן זכויות העובד בפיטורים?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.7, top_p=0.9, repetition_penalty=1.15)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Example Outputs

**Q: מהן זכויות השוכר לפי חוק השכירות?**

> לפי חוק השכירות, שוכר דירה למגורים נהנה ממספר הגנות: אסור למשכיר לדרוש בטוחות מעבר לשטר חוב וערבות בנקאית; המשכיר אינו רשאי לעקל את הערבויות שניתנו לו לפני תחילת החוק; על המשכיר למסור את הדירה כשהיא ראויה למגורים.

**Q: מה קורה אם מעסיק לא משלם פיצויי פיטורים?**

> אם המעסיק אינו משלם את הפיצויים, העובד יכול לתבוע אותו בבית הדין לעבודה. בית הדין רשאי לחייב את המעסיק בתשלום מלוא הפיצויים בצירוף הפרשי הצמדה וריבית. בנוסף ניתן להגיש תלונה ליחידת האכיפה של חוקי העבודה.

**Q: האם ניתן לערער על החלטת בית משפט השלום?**

> כן, ניתן להגיש ערעור לבית המשפט המחוזי. עם זאת, לא כל החלטה של בית משפט שלום מהווה עילה אוטומטית למתן רשות ערעור.

## Training Pipeline

```
dicta-il/dictalm2.0 (Base, 7B)
        │
        ▼
Phase 1: Continued Pretraining ── 140K legal docs ── 8,785 steps ── loss 0.850→0.700
        │
        ▼
Phase 2: Instruction Tuning ── 7,291 Q&A pairs ── 912 steps ── loss 1.63→0.87
        │
        ▼
Israeli Law Chat Model + GGUF for Ollama
```

## Training Data

| Source | Documents | Description |
|--------|:---------:|-------------|
| Israeli Courts (court.gov.il) | ~97,000 | Supreme Court and district court rulings |
| Kol-Zchut (kolzchut.org.il) | ~5,300 | Citizens' rights guides and legal explainers |
| Hebrew Wikisource | ~3,800 | Israeli legislation and Basic Laws |
| **Q&A pairs (Phase 2)** | **7,291** | Hebrew legal question-answer pairs |

## Training Results

| | Phase 1 (Pretraining) | Phase 2 (Instruction) |
|--|:-----:|:-----:|
| Data | 140K docs | 7,291 Q&A pairs |
| Steps | 8,785 | 912 |
| Final loss | 0.700 (train) / 0.769 (val) | 0.876 (train) |
| LoRA rank | 64 | 32 |
| Learning rate | 2e-4 | 1e-4 |
| Epochs | 1 | 2 |
| Time | ~7.75 hours | ~20 minutes |
| GPU | A100 40GB | A100 40GB |

## Models

| Model | Description | Link |
|-------|-------------|------|
| **Instruct (Chat)** | Ask legal questions in Hebrew | [HuggingFace](https://huggingface.co/mufeedh28/dictalm2-israeli-law-instruct-merged) |
| **GGUF** | Run locally with Ollama (F16, ~14.5 GB) | [HuggingFace](https://huggingface.co/mufeedh28/dictalm2-israeli-law-GGUF) |
| Pretrain (Base) | Text completion only, no chat | [HuggingFace](https://huggingface.co/mufeedh28/dictalm2-israeli-law-pretrain-merged) |
| Dataset | Full training data | [HuggingFace](https://huggingface.co/datasets/mufeedh28/israeli-law-pretrain) |

## Project Structure

```
scraper/              # Web scrapers for Israeli legal sources
  court.py            #   court.gov.il - Supreme & district court rulings
  kolzchut.py         #   kolzchut.org.il - Citizens' rights guides
  wikisource.py       #   Hebrew Wikisource - Israeli legislation
  knesset.py          #   Knesset API - Parliamentary data
pipeline/             # Data processing pipeline
  clean.py            #   Text cleaning & normalization
  pii.py              #   PII scrubbing (IDs, phones, emails)
  filter.py           #   Quality filtering (length, Hebrew ratio, repetition)
  dedup.py            #   Near-duplicate removal (MinHash LSH)
  format.py           #   Training data formatting & eval split
  analyze.py          #   Corpus statistics
train_dictalm.ipynb   # Phase 1: Continued pretraining (Colab)
train_instruct.ipynb  # Phase 2: Instruction tuning (Colab)
generate_qa.py        # Q&A pair generation from legal docs
run_pipeline.py       # CLI entry point for data pipeline
run_scraper.py        # CLI entry point for scrapers
```

## Reproduce

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Scrape data
python run_scraper.py

# 3. Run cleaning pipeline
python run_pipeline.py all --no-ner

# 4. Generate Q&A pairs
set GEMINI_API_KEY=your_key
python generate_qa.py

# 5. Train Phase 1 (Colab A100)
# Upload train_dictalm.ipynb to Colab and run all cells

# 6. Train Phase 2 (Colab A100)
# Upload train_instruct.ipynb to Colab and run all cells
```

## Disclaimer

This model is for research and educational purposes. It may produce inaccurate legal information. **Do not use as a substitute for professional legal advice.**

## License

Apache 2.0

## Author

[Mufeed Hammud](https://www.linkedin.com/in/mufeed-hammud-a41b84245)
