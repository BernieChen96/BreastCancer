# Requirements

# Datasets

**BreakHis** - Breast Cancer Histopathology WSI of several magnifications. Link

https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/  
Details from BreakHis website: The Breast Cancer Histopathological Image Classification (BreakHis) is composed of
9,109
microscopic images of breast tumor tissue collected from 82 patients using different magnifying factors (40X, 100X,
200X, and 400X). To date, it contains 2,480 benign and 5,429 malignant samples (700X460 pixels, 3-channel RGB, 8-bit
depth in each channel, PNG format). This database has been built in collaboration with the P&D Laboratory –
Pathological
Anatomy and Cytopathology, Parana, Brazil (http://www.prevencaoediagnose.com.br). We believe that researchers will
find
this database a useful tool since it makes future benchmarking and evaluation possible.

**Description of acronyms:**

| B = Benign           | M  = Malignant           |
|----------------------|--------------------------|
| A = Adenosis         | DC = Ductal Carcinoma    |
| F = Fibroadenoma     | LC = Lobular Carcinoma   |
| TA = Tubular Adenoma | MC = Mucinous Carcinoma  |
| PT = Phyllodes Tumor | PC = Papillary Carcinoma |

**Distribution of images in BreakHis datasets:**

<table>
    <tr>
        <td>Type</td>
        <td>Sub-category</td>
        <td>Total</td>
        <td>Patients</td>
    </tr>
    <tr>
        <td rowspan="4">Benign</td>
        <td>A</td>
        <td>444</td>
        <td rowspan="4">24</td>
    </tr>
    <tr>
        <td>F</td>
        <td>1014</td>
    </tr>
    <tr>
        <td>PT</td>
        <td>453</td>
    </tr>
    <tr>
        <td>TA</td>
        <td>569</td>
    </tr>
    <tr>
        <td rowspan="4">Malignant</td>
        <td>DC</td>
        <td>3451</td>
        <td rowspan="4">58</td>
    </tr>
    <tr>
        <td>LC</td>
        <td>626</td>
    </tr>
    <tr>
        <td>MC</td>
        <td>792</td>
    </tr>
    <tr>
        <td>PC</td>
        <td>560</td>
    </tr>
    <tr>
        <td>Total</td>
        <td> </td>
        <td>7909</td>
        <td>82</td>
    </tr>
</table>

# Commands  

### Data preparation  

**BreakHis dataset**  
```python prepare_data.py --dataset BreakHis --manual_seed 42```

### Classify  
```python classify.py --dataset BreakHis --manual_seed 42 --model efficientnet-b0```

### Metrics  
**Accuracy, Recall, F1-score, ROC, Confusion-metrix**
```python run_metrics.py --dataset BreakHis --manual_seed 42 --model efficientnet-b0```  
```python run_metrics.py --dataset BreakHis --manual_seed 42 --model densenet201```
**FID, Inception**

# Results

DenseNet201  

**Type**  

|     | precision | recall | f1-score | support |
|-----|-----------|--------|----------|---------|
| B   | 0.9       | 0.67   | 0.76     | 433     |
| M   | 0.88      | 0.97   | 0.93     | 1125    |

**Category**

|     | precision | recall | f1-score | support |
|-----|-----------|--------|----------|---------|
| A   | 0.52      | 0.70   | 0.60     | 121     |
| DC  | 0.56      | 0.77   | 0.65     | 714     |
| F   | 0.40      | 0.18   | 0.25     | 188     |
| LC  | 0.00      | 0.00   | 0.00     | 103     |
| MC  | 0.30      | 0.28   | 0.29     | 181     |
| PC  | 0.00      | 0.00   | 0.00     | 127     |
| PT  | 0.00      | 0.00   | 0.00     | 60      |
| TA  | 0.08      | 0.08   | 0.08     | 64      |