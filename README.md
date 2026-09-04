# LLM Fine-Tuning with LoRA on Google Colab — Text-to-SQL

Fine-tunes **TinyLlama-1.1B-Chat** with **LoRA** to turn `(table schema, question)` pairs into clean SQL queries, runnable end-to-end on a free Colab T4 GPU in under 15 minutes.

The base model already "knows" SQL, but answers like a chatbot — explaining the query, wrapping it in markdown, adding commentary. This project isn't about teaching SQL syntax; it's about teaching a **response format**: one query, no fluff, ready to execute programmatically. LoRA is used instead of full fine-tuning because updating all 1.1B parameters isn't necessary for a format shift, and isn't feasible on 16GB of VRAM anyway.

## What's in this repo

| File | Description |
|---|---|
| `LLM_Fine_Tuning_with_LoRA_on_Google_Colab_for_Text_to_SQL_IMPROVED.ipynb` | The full notebook — run this top to bottom in Colab |

## Requirements

- Google Colab with a **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU). Free tier is sufficient.
- No local setup needed — the first cell installs everything (`transformers`, `peft`, `accelerate`, `datasets`, `trl`, `sentencepiece`, `protobuf`) inside the Colab environment.
- No API keys — the base model and dataset are pulled from the Hugging Face Hub anonymously.

## How to run

1. Open the notebook in Colab (upload it, or `File → Upload notebook`).
2. Set the runtime to T4 GPU.
3. Run all cells in order (`Runtime → Run all`). Total time: ~10–15 minutes, mostly the training step.
4. Read the printed BEFORE/AFTER probe comparisons and the exact-match accuracy at the end.

## What the notebook does

1. **Environment setup** — installs dependencies, checks GPU, fixes a random seed for reproducibility.
2. **Loads the base model** in FP16 (~2.2 GB VRAM).
3. **Probes the base model** on 3 held-out questions to show its default (chatty, unstructured) behavior.
4. **Loads and formats data** — 3,000 examples from [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context), formatted with TinyLlama's chat template. A raw (unformatted) copy of the eval split is kept aside for scoring later.
5. **Attaches LoRA adapters** (rank 16) on both attention (`q/k/v/o_proj`) and MLP (`gate/up/down_proj`) projections — under 1% of parameters become trainable.
6. **Trains** for 1 epoch via `SFTTrainer`, with the best checkpoint (by eval loss) reloaded at the end.
7. **Re-probes** the same 3 questions post-training for a direct before/after comparison.
8. **Measures exact-match accuracy** on the full held-out set — a real correctness check, not just a loss number.
9. **Saves the LoRA adapter** (~20 MB) — the base model itself is never re-saved.

## Results you should see

- **Peak VRAM:** ~4 GB (well under the T4's 16 GB)
- **Training time:** ~10 minutes for 1 epoch / 3,000 examples
- **Before fine-tuning:** verbose, multi-sentence answers with markdown code fences and re-explained schemas
- **After fine-tuning:** a single, clean SQL statement, e.g. `SELECT name FROM employees WHERE department = "engineering" AND salary > 100000`
- **Adapter size:** ~20 MB on disk

## Known limitations

This is a demonstration of the LoRA *workflow*, not a production SQL model:

- 3,000 training examples (out of ~78k available) and 1 epoch are enough to shift **response style**, not to guarantee **semantic correctness** on every query.
- Evaluation is **exact-match** (case-insensitive, whitespace-normalized string comparison), which is strict — semantically identical SQL with different formatting or aliasing will count as a miss. Execution-based evaluation (running both queries against a real schema and diffing results) would be a fairer but heavier next step.
- Decoding is greedy (`do_sample=False`) for reproducibility, not tuned for production serving.

## Scaling this up

The workflow doesn't change, only the knobs:

- Swap in the full dataset (~78k examples) instead of the 3k slice.
- Train 2–3 epochs instead of 1.
- Move to a larger base model (e.g. Qwen2.5-3B, or a 7B+ model with **QLoRA** 4-bit quantization to keep it fitting in T4 memory).
- Add execution-based accuracy scoring alongside exact-match.

## Loading the saved adapter later

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./tinyllama-sql-lora-adapter")
tokenizer = AutoTokenizer.from_pretrained("./tinyllama-sql-lora-adapter")
```

## Credits

- Base model: [`TinyLlama/TinyLlama-1.1B-Chat-v1.0`](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- Dataset: [`b-mc2/sql-create-context`](https://huggingface.co/datasets/b-mc2/sql-create-context)
- Libraries: Hugging Face `transformers`, `peft`, `trl`, `datasets`, `accelerate`
