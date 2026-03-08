# Israeli Law LLM

A Hebrew legal language model fine-tuned on 140,000+ Israeli legal documents. Built on [DictaLM 2.0](https://huggingface.co/dicta-il/dictalm2.0) (7B parameters, Mistral-based).

**Model on HuggingFace:** [mufeedh28/dictalm2-israeli-law-merged](https://huggingface.co/mufeedh28/dictalm2-israeli-law-merged)

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mufeedh28/dictalm2-israeli-law-merged")
tokenizer = AutoTokenizer.from_pretrained("mufeedh28/dictalm2-israeli-law-merged")

inputs = tokenizer("בית המשפט העליון פסק כי", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Project Structure

```
scraper/          # Web scrapers for Israeli legal sources
  court.py        # court.gov.il - Supreme & district court rulings
  kolzchut.py     # kolzchut.org.il - Citizens' rights guides
  wikisource.py   # Hebrew Wikisource - Israeli legislation
  knesset.py      # Knesset API - Parliamentary data
pipeline/         # Data processing pipeline
  clean.py        # Text cleaning & normalization
  pii.py          # PII scrubbing (IDs, phones, emails)
  filter.py       # Quality filtering (length, Hebrew ratio, repetition)
  dedup.py        # Exact + near-duplicate removal (MinHash LSH)
  format.py       # Training data formatting & eval split
  analyze.py      # Corpus statistics & tokenizer analysis
run_pipeline.py   # CLI entry point for the full pipeline
run_scraper.py    # CLI entry point for scrapers
train_dictalm.ipynb  # Colab training notebook (Unsloth + QLoRA)
```

## Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Scrape data

```bash
python run_scraper.py
```

### 3. Run cleaning pipeline

```bash
python run_pipeline.py all --no-ner
```

### 4. Train on Colab

Upload `train_dictalm.ipynb` to Google Colab with an A100 GPU and run all cells.

## Training Data

| Source | Documents | Description |
|--------|-----------|-------------|
| Israeli Courts | ~97,000 | Supreme Court and district court rulings |
| Kol-Zchut | ~5,300 | Citizens' rights guides |
| Wikisource | ~3,800 | Israeli legislation and basic laws |

## Training Results

| Metric | Value |
|--------|-------|
| Final training loss | 0.700 |
| Final validation loss | 0.769 |
| Training time | ~7.75 hours (A100) |
| Trainable parameters | 167M / 7.4B (2.26%) |

## License

Apache 2.0
