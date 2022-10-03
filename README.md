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

Data preparation

**BreakHis dataset**  
```python -m data.prepare_data```
     