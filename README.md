# Amharic Next-Word Suggestion — Streamlit starter

This is the Streamlit version of the notebook's Task 4 interface. It uses the saved n-gram tables; it does **not** retrain the model.

## 1. Expected files

Create a `model/` folder beside `app.py` and put these eight arrays plus the two JSON files in it:

```text
model/
├── vocab.json
├── ngram_totals.json
├── ngram_keys_1.npy
├── ngram_counts_1.npy
├── ngram_keys_2.npy
├── ngram_counts_2.npy
├── ngram_keys_3.npy
├── ngram_counts_3.npy
├── ngram_keys_4.npy
└── ngram_counts_4.npy
```

These filenames match the notebook. The app expects the **full model with stopwords**, as used in the notebook's final Task 4 app.

## 2. Run locally

In PowerShell, from this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use `\.venv\Scripts\python.exe -m pip install -r requirements.txt` and `\.venv\Scripts\python.exe -m streamlit run app.py`.

## 3. Deploy

Push `app.py`, `requirements.txt`, and this README to the root of the GitHub repository. Then open https://share.streamlit.io/, choose **Create app**, select the repository and branch, and set the main file path to `app.py`.

Do not commit the large `.npy` model files to ordinary Git history. The model arrays are very large, so a separate artifact host or a platform with more memory may be needed.

## Important hosting limitation

The notebook reports a vocabulary of about 772,540 tokens and model arrays totaling roughly 1.16 GB before context statistics and runtime overhead. Streamlit Community Cloud's documented memory ceiling is approximately 2.7 GB and varies by app. Therefore, this full model may exceed the free host's memory budget even with memory-mapped arrays. The app is prepared, but **successful public deployment is not guaranteed** until the model loads within the selected host's limits. If it fails, use a host with more RAM or build a smaller/pruned model artifact; do not rerun training just to deploy this interface.
