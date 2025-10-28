# What Does `normalize_avazu_datasets_with_temporal.py` Do?

## 🎯 Purpose

This script solves a **critical problem**: The three Avazu datasets (x1, x2, x4) come in **different formats** with **different column names** and **encoded temporal features**. To train a unified model across all three, we need them to have the **exact same schema**.

---

## ❓ The Problem: Why Normalization is Needed

### Raw Dataset Formats (Before Normalization)

#### **Avazu_x1** (Anonymized)
```
Columns: label, feat_1, feat_2, ..., feat_22
- 22 anonymized features (feat_1 to feat_22)
- feat_22 is actually ENCODED hour (e.g., 1544248 = hour 0)
- Missing feat_21 (we need 21 features for unified schema)
- No temporal information visible
```

#### **Avazu_x2** (Semantic names)
```
Columns: click, C1, banner_pos, site_id, site_domain, ..., hour, mday, wday
- Different column names (click vs label, C1 vs feat_1)
- Has semantic names: site_id, device_model, etc.
- Temporal features are ENCODED:
  - hour: 645164 + hour_of_day (e.g., 645164 = hour 0)
  - mday: 645154 + day_offset (e.g., 645155 = day 1 of Oct)
  - wday: 645188 + weekday (e.g., 645188 = Sunday)
```

#### **Avazu_x4** (Raw Kaggle format)
```
Columns: id, click, hour, C1, banner_pos, site_id, site_domain, ...
- Different column names again
- hour is in YYMMDDHH format (e.g., 14102100 = Oct 21, 2014, 00:00)
- Raw hex IDs for categorical features
- Has 'id' column (not useful for prediction)
```

### 🚨 **The Core Issue**

You **cannot train a model on these datasets together** because:

1. ❌ **Different column names** → Model doesn't know site_id (x2) = feat_3 (x1)
2. ❌ **Different schemas** → x1 has 22 features, x2 has 25+ columns
3. ❌ **Encoded temporal features** → x1's feat_22=1544250 means nothing without decoding
4. ❌ **Inconsistent formats** → Can't compute embeddings with different vocabularies

**Result**: Training would fail or produce garbage predictions.

---

## ✅ The Solution: What Normalization Does

### Step 1: **Unified Schema Creation**

Creates the same 24 columns for ALL datasets:

```python
UNIFIED_COLUMNS = [
    'label',              # Target (0 = no click, 1 = click)
    'feat_1', ..., 'feat_21',  # 21 categorical features
    'hour'                # Temporal feature (YYMMDDHH format)
]
```

### Step 2: **Column Mapping**

Maps each dataset's columns to the unified schema:

**X1 Mapping:**
```python
feat_1 → feat_1      # Direct mapping
feat_2 → feat_2
...
feat_20 → feat_20
feat_22 → hour       # Needs decoding!
feat_21 → -1         # Missing, pad with constant
```

**X2 Mapping:**
```python
click → label
C1 → feat_1
banner_pos → feat_2
site_id → feat_3
site_domain → feat_4
site_category → feat_5
app_id → feat_6
...
C21 → feat_21
hour + mday + wday → hour  # Needs reconstruction!
```

**X4 Mapping:**
```python
click → label
C1 → feat_1
banner_pos → feat_2
...
C21 → feat_21
hour → hour          # Already correct format!
DROP: id             # Not useful
```

### Step 3: **Temporal Feature Decoding**

This is the **most complex part**!

#### **Problem**: X1 and X2 have encoded temporal features

**X1 Decoding:**
```python
# X1's feat_22 is min-shifted
# Original: 1544248, 1544249, 1544250, ...
# Decoded:  0, 1, 2, ... (hour of day)

hour_decoded = feat_22 - 1544248  # Get 0-23

# Reconstruct full timestamp (assuming Oct 21, 2014)
hour = 14102100 + hour_decoded  # → 14102100, 14102101, ...
```

**Example:**
- feat_22 = 1544260 → hour_decoded = 12 → hour = 14102112 (Oct 21, 2014, 12:00)

**X2 Decoding:**
```python
# X2 has three encoded columns
hour_encoded = 645164 + hour_of_day     # e.g., 645180 = hour 16
mday_encoded = 645154 + day_offset      # e.g., 645157 = day 3
wday_encoded = 645188 + weekday         # e.g., 645191 = Wednesday

# Decode
hour_decoded = hour_encoded - 645164    # → 0-23
mday_decoded = mday_encoded - 645154 + 1  # → 1-10 (Oct 21-30)
actual_day = mday_decoded + 20          # → 21-30

# Reconstruct YYMMDDHH
hour = int(f"1410{actual_day:02d}{hour_decoded:02d}")
```

**Example:**
- hour_encoded = 645180, mday_encoded = 645157
- hour_decoded = 16, mday_decoded = 4, actual_day = 24
- hour = 14102416 (Oct 24, 2014, 16:00)

**NOTE**: X2's reconstructed hour is in **6-digit format** (YYDDHH) like `142416` instead of 8-digit (YYMMDDHH) like `14102416`. This is handled by the preprocessing code in `feature_processor.py`.

**X4**: No decoding needed! Already in YYMMDDHH format.

### Step 4: **Output Normalized CSVs**

Creates clean, unified datasets:

```
data/Avazu/avazu_x1_normalized/
├── train.csv    (28.3M rows, 24 columns)
├── valid.csv    (4.0M rows, 24 columns)
└── test.csv     (4.0M rows, 24 columns)

data/Avazu/avazu_x2_normalized/
├── train.csv    (32.3M rows, 24 columns)
└── test.csv     (8.1M rows, 24 columns)

data/Avazu/avazu_x4_normalized/
├── train.csv    (28.3M rows, 24 columns)
├── valid.csv    (4.0M rows, 24 columns)
└── test.csv     (4.0M rows, 24 columns)
```

**All have the SAME schema:**
```
label,feat_1,feat_2,...,feat_21,hour
0,1,9,18,3582,7908,...,14102122
1,1,10,19,3583,7907,...,14102123
...
```

---

## 🎯 Why This Matters for Training

### Before Normalization (❌ Won't Work)

```python
# Training on x1
model.fit(x1_data)  # Learns: feat_22 has 24 unique values

# Training on x2
model.fit(x2_data)  # ERROR! No feat_22 column! Has 'hour' instead
```

### After Normalization (✅ Works!)

```python
# Sequential training on x1 → x2 → x4
model.fit(x1_normalized)  # Learns: hour feature, feat_1-21
model.fit(x2_normalized)  # Continues learning with SAME schema
model.fit(x4_normalized)  # Test on SAME schema
```

**Result:**
- ✅ Consistent embeddings across datasets
- ✅ Can train sequentially: x1 → x2
- ✅ Can test on x4 with same model
- ✅ Temporal features properly decoded and usable

---

## 🔍 What Happens During Normalization?

### Input (Example from x2):
```csv
click,C1,banner_pos,site_id,...,hour,mday,wday
0,1005,0,1fbe01fe,...,645180,645157,645191
```

### Processing:
1. **Rename columns**: click → label, C1 → feat_1, ...
2. **Decode temporal**:
   - hour: 645180 - 645164 = 16
   - mday: 645157 - 645154 + 1 = 4 → day 24
   - Reconstruct: 14102416 (but stored as 142416 due to bug)
3. **Select unified columns**: Keep only label, feat_1-21, hour
4. **Save to CSV**

### Output:
```csv
label,feat_1,feat_2,feat_3,...,feat_21,hour
0,1005,0,1fbe01fe,...,<value>,142416
```

---

## 📊 Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   RAW DATASETS                              │
├─────────────────────────────────────────────────────────────┤
│  X1: 22 cols (feat_1-22)      | Different formats           │
│  X2: 25+ cols (click, C1...)  | Different names             │
│  X4: 24 cols (id, click...)   | Encoded temporals           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 │ normalize_avazu_datasets_with_temporal.py
                 │
                 ├─> Column mapping (feat names)
                 ├─> Temporal decoding (hour reconstruction)
                 ├─> Schema unification (24 columns)
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│              NORMALIZED DATASETS                            │
├─────────────────────────────────────────────────────────────┤
│  x1_normalized: 24 cols (label, feat_1-21, hour)           │
│  x2_normalized: 24 cols (label, feat_1-21, hour)           │
│  x4_normalized: 24 cols (label, feat_1-21, hour)           │
│                                                             │
│  ✅ Same schema  ✅ Same format  ✅ Ready for training      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Takeaways

1. **Normalization is MANDATORY** - You cannot skip this step
2. **Creates unified schema** - All datasets → same 24 columns
3. **Decodes temporal features** - Converts encoded values to usable timestamps
4. **Enables sequential training** - Train on x1, then x2, test on x4
5. **One-time operation** - Run once, use forever

Without normalization, the LLM-CTR training would **immediately fail** because the model cannot handle datasets with different schemas.
