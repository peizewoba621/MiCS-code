# MiCS

Code for **MiCS** (Mechanistic Interpretability-based Context Selection). Candidate paragraphs are scored from copy-head attention and hidden-state cosine similarity; the main tables keep the top two paragraphs (\(k=2\)).

This folder is the full release. Model weights and JSONL datasets are not included.

## Files

| File | Role |
| --- | --- |
| `hp_token_level_detect.py` | Score each of the ten candidates; rank by Ave external similarity; keep top 2 |
| `hp_token_level_detect2.py` | Score pairwise candidate combinations and keep the best pair |
| `simple_llama2_test.py` | End-to-end generation and EM / Token-F1 / Rouge-L / AiC for Truth, All contexts, Random, TF-IDF, BM25, Dense, Reordering, Hybrid |
| `simple_llama2_test2.py` | Smaller generation/eval helper |
| `inference_aggregate.py` | Optional context-aggregation inference after ranking |
| `test_llama2_7B/topk_heads.json` | Copy-head list for Llama-2-7B (32 heads) |
| `test_llama2_13B/topk_heads.json` | Copy-head list for Llama-2-13B (40 heads) |
| `test_llama3_8B/topk_heads.json` | Copy-head list for Llama-3-8B (32 heads) |

## Requirements

Python 3.9+, a CUDA GPU, and:

```bash
pip install torch transformers numpy scikit-learn sentence-transformers tqdm
```

Edit the hardcoded checkpoint strings (currently under `/hd/pengzewei/...`) to your local Hugging Face paths, for example `meta-llama/Llama-2-7b-chat-hf`. Dense retrieval uses `sentence-transformers/all-MiniLM-L6-v2`. Reordering uses BM25 then `cross-encoder/ms-marco-MiniLM-L-6-v2`.

The scoring scripts load heads from `./log/test_llama2_7B/topk_heads.json` (and the 13B / Llama-3 counterparts). The JSON files in this folder are `test_llama2_7B/topk_heads.json` etc. Copy or symlink them into a `log/` layout, or change the paths in the scripts.

## Data (not in this folder)

This release contains **code and `topk_heads.json` only**. Question files, converted JSONL, and model weights are **not** included.

The scripts still have hardcoded local paths from the original runs. Point them at your own files before running:

| Script | What to edit |
| --- | --- |
| `hp_token_level_detect.py` | `response_path` / `source_info_path` (`hotpot_converted_10.jsonl`, `wk_converted_10.jsonl`, `mu_converted_10.jsonl`) |
| `hp_token_level_detect2.py` | same JSONL paths |
| `simple_llama2_test.py` | `hotpot_data_path`, `aggregate_results_path`, and the Llama checkpoint string |
| `simple_llama2_test2.py` | same |
| `inference_aggregate.py` | input JSON / output JSONL paths |

Expected JSONL fields (one example per line): `question`, `answer`, `supporting_facts`, `context0` … `context9`.

Official sources: [HotpotQA](https://hotpotqa.github.io/), [2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), [MuSiQue](https://github.com/StonyBrookNLP/musique).

## Defaults in the paper

- Selective methods use \(k=2\)
- Top 10% prefix tokens per copy head (mean-pooled)
- Ranking score: average of token-level \(\mathrm{cos}(h_t, e_t)\) over generated tokens and selected heads (Ave)
- BM25: \(k_1=1.5\), \(b=0.75\); Hybrid: 0.5 BM25 + 0.5 MiniLM after min-max

## Score candidates (MiCS)

```bash
python hp_token_level_detect.py --model_name llama2-7b --dataset hotpot
```

`--model_name`: `llama2-7b`, `llama2-13b`, or `llama3-8b`. `--dataset`: `hotpot`, `wiki`, or `mu`.

For each generated token the loop takes copy-head attention over that candidate’s **prefix**, mean-pools the top 10% prefix states to \(e_t\), computes \(\mathrm{cos}(h_t,e_t)\), and averages over tokens and heads.

Pairwise variant:

```bash
python hp_token_level_detect2.py --model_name llama2-7b --dataset hotpot
```

## End-to-end metrics

```bash
python simple_llama2_test.py
```

Then choose:

1. Truth (gold supporting paragraphs)
2. All contexts
3. Two random paragraphs
4. Top-3 from stored scores
5. TF-IDF top-2
6. BM25 top-2
7. Dense top-2
8. Reordering (BM25 top-10, then cross-encoder top-2)
9. Hybrid (0.5 / 0.5)

Exact Match is 1 iff the normalized prediction equals the single gold answer string; the reported EM is the mean over examples.

The HotpotQA case study in the paper is the first converted item (`Were Scott Derrickson and Ed Wood of the same nationality?`), gold idx 1 and 4.

## License

Research code released with the paper. Datasets and model weights remain under their original licenses.
