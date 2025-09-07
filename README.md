# Home Credit Default Risk — LightGBM

LighGBM for Kaggle's **Home Credit Default Risk**

Got 0.792 private and 0.794 public ROC-AUC on Kaggle

**Quickstart**
```powershell
cd D:\home_credit
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r D:\home_credit\requirements.txt

Put Kaggle CSVs into: D:\home_credit\data\raw

python D:\home_credit\train_hcdr_lgbm.py --data-dir D:\home_credit\data\raw --out-dir D:\home_credit\outputs `
  --folds 5 --early-stopping 300 --seeds 52 42