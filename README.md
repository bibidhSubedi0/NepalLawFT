```markdown
# NepalLawFT

QLoRA fine-tune of Llama-3.2-3B-Instruct on a domain-specific Nepali legal Q&A dataset. The model is trained to answer questions about Nepal's laws, constitution, and governance documents in both Nepali and English, with source citations.

## Results

Evaluated on 50 held-out test samples against the base model.

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| ROUGE-L | 0.1975 | 0.2913 | +47.5% |
| BLEU (char bigram) | 0.3827 | 0.4798 | +25.4% |
| Semantic Similarity | 0.5400 | 0.6823 | +26.4% |
| LLM Judge (1-5) | 1.720 | 2.600 | +51.2% |

LLM-as-Judge scoring via Groq (`llama-3.3-70b-versatile`).

## Training Setup

| Parameter | Value |
|---|---|
| Base model | meta-llama/Llama-3.2-3B-Instruct |
| Method | QLoRA (4-bit NF4) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down proj |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Max sequence length | 512 |
| Trainable parameters | ~24M (~1% of base) |

## Dataset

Domain-specific Nepali legal Q&A pairs sourced from Nepal's constitution, acts, and governance documents. Mixed Nepali and English.

| Split | Samples |
|---|---|
| Train | ~3,200 |
| Validation | ~400 |
| Test | ~430 |

## Structure

```
├── notebooks/
│   ├── 00_data_split.ipynb       # Train/val/test split
│   ├── 01_finetune.ipynb         # QLoRA fine-tuning
│   └── 02_eval.ipynb             # Evaluation pipeline
├── scripts/
│   ├── 01_generate_dataset.py    # Dataset generation
│   └── chunk_audit.py            # Chunk quality audit
├── space/
│   ├── app.py                    # Gradio demo
│   └── requirements.txt
└── results/                      # Eval outputs and summary
```

## Links

- Adapter: [Bibidh/civicLens-llama3.2-3b-nepali-legal](https://huggingface.co/Bibidh/civicLens-llama3.2-3b-nepali-legal)
- Merged model: [Bibidh/civicLens-llama3.2-3b-nepali-legal-merged](https://huggingface.co/Bibidh/civicLens-llama3.2-3b-nepali-legal-merged)
- Demo: [Bibidh/NepalLawFT-Demo](https://huggingface.co/spaces/Bibidh/NepalLawFT-Demo)
```