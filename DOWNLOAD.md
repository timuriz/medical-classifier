# Medical Image Classifier - Download & Setup Guide

## 📥 What to Download

### Option 1: Full HAM10000 Dataset (RECOMMENDED)

**Dataset:** https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

**Download Size:** ~3.2 GB

**Steps:**
1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Click "Download" (requires free Kaggle account)
3. Extract the ZIP file
4. Copy the entire folder to: `data/raw_images/`

**Expected folder structure after extraction:**
```
data/
  └── raw_images/
      ├── HAM10000_images_part1/
      ├── HAM10000_images_part2/
      ├── HAM10000_metadata.csv
      ├── README.md
      └── ...
```

### Option 2: Quick Setup with Script

Once you have the dataset extracted:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Organize images by class
python reorganize_dataset.py

# 3. This will create:
#    data/melanoma/
#    data/nevus/
#    data/seborrheic_keratosis/
```

---

## 🚀 Complete Workflow

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle (manual - 3.2GB)
# Extract to data/raw_images/

# 4. Organize by class
python reorganize_dataset.py

# 5. Test everything works
python backend/train/dataset.py

# 6. Train model
python backend/train/train_model.py
```

---

## 📊 Project Structure

```
medical-classifier/
├── requirements.txt              ← Dependencies to install
├── reorganize_dataset.py         ← Organize images by class
├── README.md
│
├── backend/train/
│   ├── dataset.py               ← PyTorch Dataset class
│   └── train_model.py           ← Training script
│
├── models/                       ← Trained weights saved here
│
└── data/
    ├── raw_images/              ← Download HAM10000 here
    ├── melanoma/                ← After organizing
    ├── nevus/
    └── seborrheic_keratosis/
```

---

## ⏱️ Expected Training Time

- **GPU (RTX 3060+):** ~45 minutes
- **CPU:** ~3-4 hours
- **Google Colab (free GPU):** ~1-2 hours

---

## 🎯 Dataset Info

**HAM10000 Contains:**
- 10,015 dermatoscopic images
- 7 types of skin lesions
- High quality labeled data

**We Use 3 Classes:**
- Melanoma (1,113 images)
- Nevus (6,705 images)
- Seborrheic Keratosis (2,197 images)

**Expected Accuracy:**
- Train: ~85%
- Validation: ~80%
- Test: ~78%

---

## 💻 System Requirements

- Python 3.9+
- 4GB RAM minimum (8GB+ recommended)
- 5GB disk space (for dataset + models)
- GPU optional but recommended

---

## ✅ Next Steps

1. Download HAM10000 from Kaggle
2. Extract to `data/raw_images/`
3. Run `python reorganize_dataset.py`
4. Run `python backend/train/train_model.py`

You're all set! 🚀
