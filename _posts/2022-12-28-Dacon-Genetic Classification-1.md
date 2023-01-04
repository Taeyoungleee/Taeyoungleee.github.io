```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'sudo apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 20 not upgraded.
    /usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
    /usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
    /usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
    /usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
    /usr/share/fonts/truetype/nanum: caching, new cache contents: 10 fonts, 0 dirs
    /usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
    /root/.local/share/fonts: skipping, no such directory
    /root/.fonts: skipping, no such directory
    /var/cache/fontconfig: cleaning cache directory
    /root/.cache/fontconfig: not cleaning non-existent cache directory
    /root/.fontconfig: not cleaning non-existent cache directory
    fc-cache: succeeded



```python
pip install catboost
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: catboost in /usr/local/lib/python3.8/dist-packages (1.1.1)
    Requirement already satisfied: graphviz in /usr/local/lib/python3.8/dist-packages (from catboost) (0.10.1)
    Requirement already satisfied: numpy>=1.16.0 in /usr/local/lib/python3.8/dist-packages (from catboost) (1.21.6)
    Requirement already satisfied: pandas>=0.24.0 in /usr/local/lib/python3.8/dist-packages (from catboost) (1.3.5)
    Requirement already satisfied: scipy in /usr/local/lib/python3.8/dist-packages (from catboost) (1.7.3)
    Requirement already satisfied: plotly in /usr/local/lib/python3.8/dist-packages (from catboost) (5.5.0)
    Requirement already satisfied: six in /usr/local/lib/python3.8/dist-packages (from catboost) (1.15.0)
    Requirement already satisfied: matplotlib in /usr/local/lib/python3.8/dist-packages (from catboost) (3.2.2)
    Requirement already satisfied: python-dateutil>=2.7.3 in /usr/local/lib/python3.8/dist-packages (from pandas>=0.24.0->catboost) (2.8.2)
    Requirement already satisfied: pytz>=2017.3 in /usr/local/lib/python3.8/dist-packages (from pandas>=0.24.0->catboost) (2022.7)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->catboost) (3.0.9)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib->catboost) (1.4.4)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.8/dist-packages (from matplotlib->catboost) (0.11.0)
    Requirement already satisfied: tenacity>=6.2.0 in /usr/local/lib/python3.8/dist-packages (from plotly->catboost) (8.1.0)



```python
pip install xgboost
```

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Requirement already satisfied: xgboost in /usr/local/lib/python3.8/dist-packages (0.90)
    Requirement already satisfied: scipy in /usr/local/lib/python3.8/dist-packages (from xgboost) (1.7.3)
    Requirement already satisfied: numpy in /usr/local/lib/python3.8/dist-packages (from xgboost) (1.21.6)



```python
import pandas as pd
import random
import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
%matplotlib inline

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
# plt.rc('font', family='AppleGothic')

import warnings
warnings.filterwarnings(action='ignore') 
```


```python
class CFG:
    SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(CFG.SEED) # Seed 고정
```


```python
import matplotlib.pyplot as plt
plt.rc('font', family='NanumBarunGothic')
```


```python
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive



```python
snp_info= pd.read_csv("/content/drive/MyDrive/유전체/snp_info.csv")
train= pd.read_csv("/content/drive/MyDrive/유전체/train.csv")
test= pd.read_csv("/content/drive/MyDrive/유전체/test.csv")
sample_submission= pd.read_csv("/content/drive/MyDrive/유전체/sample_submission.csv")
```

Dataset Info.

### train.csv [파일]
1. id : 개체 고유 ID
개체정보
2. father : 개체의 가계 고유 번호 (0 : Unknown)
3. mother : 개체의 모계 고유 번호 (0 : Unknown)
4. gender : 개체 성별 (0 : Unknown, 1 : female, 2 : male)
5. trait : 개체 표현형 정보 
(15개의 SNP 정보 : SNP_01 ~ SNP_15)
6. class : 개체의 품종 (A,B,C)


### test.csv [파일]
1. id : 개체 샘플 별 고유 ID
개체정보
2. father : 개체의 가계 고유 번호 (0 : Unknown)
3. mother : 개체의 모계 고유 번호 (0 : Unknown)
4. gender : 개체 성별 (0 : Unknown, 1 : female, 2 : male)
5. trait : 개체 표현형 정보 
(15개의 SNP 정보 : SNP_01 ~ SNP_15)


### snp_info.csv [파일]
15개의 SNP 세부 정보
1. name : SNP 명
2. chrom : 염색체 정보
3. cm : Genetic distance
4. pos : 각 마커의 유전체상 위치 정보


### sample_submission.csv [파일] - 제출 양식
1. id : 개체 샘플 별 고유 ID
2. class : 예측한 개체의 품종 (A,B,C)

# 1. 데이터 살펴보기


```python
snp_info
```





  <div id="df-fc5c7dfb-e1a8-44ce-8f8b-ce83e77af579">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SNP_01</td>
      <td>BTA-19852-no-rs</td>
      <td>2</td>
      <td>67.05460</td>
      <td>42986890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.15670</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.28920</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.87490</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.50150</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.59540</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.78000</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.68560</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.87400</td>
      <td>73092782</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SNP_10</td>
      <td>BTB-01558306</td>
      <td>7</td>
      <td>62.06920</td>
      <td>40827112</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SNP_11</td>
      <td>ARS-BFGL-NGS-44247</td>
      <td>8</td>
      <td>97.17310</td>
      <td>92485682</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.74630</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.41810</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.81970</td>
      <td>72822507</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SNP_15</td>
      <td>BovineHD1000000224</td>
      <td>10</td>
      <td>1.78774</td>
      <td>814291</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-fc5c7dfb-e1a8-44ce-8f8b-ce83e77af579')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-fc5c7dfb-e1a8-44ce-8f8b-ce83e77af579 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-fc5c7dfb-e1a8-44ce-8f8b-ce83e77af579');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
train
```





  <div id="df-43dda29d-1369-4063-b57c-b22dc76e89ef">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>...</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TRAIN_000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C A</td>
      <td>...</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TRAIN_001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>A G</td>
      <td>C A</td>
      <td>A A</td>
      <td>A A</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TRAIN_002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TRAIN_003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G G</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TRAIN_004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>TRAIN_257</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>B</td>
    </tr>
    <tr>
      <th>258</th>
      <td>TRAIN_258</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>C A</td>
      <td>A A</td>
      <td>A A</td>
      <td>...</td>
      <td>G A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>C</td>
    </tr>
    <tr>
      <th>259</th>
      <td>TRAIN_259</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>G A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>G G</td>
      <td>C A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>260</th>
      <td>TRAIN_260</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>...</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A G</td>
      <td>G A</td>
      <td>G G</td>
      <td>C A</td>
      <td>G G</td>
      <td>A</td>
    </tr>
    <tr>
      <th>261</th>
      <td>TRAIN_261</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C A</td>
      <td>G G</td>
      <td>C C</td>
      <td>...</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 21 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-43dda29d-1369-4063-b57c-b22dc76e89ef')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-43dda29d-1369-4063-b57c-b22dc76e89ef button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-43dda29d-1369-4063-b57c-b22dc76e89ef');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
test
```





  <div id="df-26504737-3f2a-43f9-a060-ecfcbac48f80">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>A G</td>
      <td>G G</td>
      <td>G A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A G</td>
      <td>G A</td>
      <td>G G</td>
      <td>C A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_001</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C C</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_002</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_003</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A G</td>
      <td>C A</td>
      <td>A A</td>
      <td>C C</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_004</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>G G</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>TEST_170</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>G G</td>
      <td>C C</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>171</th>
      <td>TEST_171</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G A</td>
    </tr>
    <tr>
      <th>172</th>
      <td>TEST_172</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>C A</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A G</td>
      <td>A A</td>
      <td>G G</td>
    </tr>
    <tr>
      <th>173</th>
      <td>TEST_173</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>A G</td>
      <td>G G</td>
      <td>C A</td>
      <td>G A</td>
      <td>C C</td>
      <td>G G</td>
      <td>A A</td>
      <td>G A</td>
      <td>A A</td>
      <td>G G</td>
      <td>A G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
    <tr>
      <th>174</th>
      <td>TEST_174</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>G G</td>
      <td>G G</td>
      <td>C C</td>
      <td>G A</td>
      <td>C A</td>
      <td>A A</td>
      <td>G A</td>
      <td>G G</td>
      <td>A A</td>
      <td>G G</td>
      <td>G G</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
      <td>A A</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 20 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-26504737-3f2a-43f9-a060-ecfcbac48f80')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-26504737-3f2a-43f9-a060-ecfcbac48f80 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-26504737-3f2a-43f9-a060-ecfcbac48f80');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
print(f"train : {train.shape}")
print(f"test : {test.shape}")
print(f"submission : {sample_submission.shape}")
```

    train : (262, 21)
    test : (175, 20)
    submission : (175, 2)



```python
snp_info.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 15 entries, 0 to 14
    Data columns (total 5 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   SNP_id  15 non-null     object 
     1   name    15 non-null     object 
     2   chrom   15 non-null     int64  
     3   cm      15 non-null     float64
     4   pos     15 non-null     int64  
    dtypes: float64(1), int64(2), object(2)
    memory usage: 728.0+ bytes



```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 262 entries, 0 to 261
    Data columns (total 21 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      262 non-null    object
     1   father  262 non-null    int64 
     2   mother  262 non-null    int64 
     3   gender  262 non-null    int64 
     4   trait   262 non-null    int64 
     5   SNP_01  262 non-null    object
     6   SNP_02  262 non-null    object
     7   SNP_03  262 non-null    object
     8   SNP_04  262 non-null    object
     9   SNP_05  262 non-null    object
     10  SNP_06  262 non-null    object
     11  SNP_07  262 non-null    object
     12  SNP_08  262 non-null    object
     13  SNP_09  262 non-null    object
     14  SNP_10  262 non-null    object
     15  SNP_11  262 non-null    object
     16  SNP_12  262 non-null    object
     17  SNP_13  262 non-null    object
     18  SNP_14  262 non-null    object
     19  SNP_15  262 non-null    object
     20  class   262 non-null    object
    dtypes: int64(4), object(17)
    memory usage: 43.1+ KB


## SNP_info 분석


```python
snp_info
```





  <div id="df-7df1db76-4200-403f-a46d-c42bd76f9e15">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SNP_01</td>
      <td>BTA-19852-no-rs</td>
      <td>2</td>
      <td>67.05460</td>
      <td>42986890</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.15670</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.28920</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.87490</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.50150</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.59540</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.78000</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.68560</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.87400</td>
      <td>73092782</td>
    </tr>
    <tr>
      <th>9</th>
      <td>SNP_10</td>
      <td>BTB-01558306</td>
      <td>7</td>
      <td>62.06920</td>
      <td>40827112</td>
    </tr>
    <tr>
      <th>10</th>
      <td>SNP_11</td>
      <td>ARS-BFGL-NGS-44247</td>
      <td>8</td>
      <td>97.17310</td>
      <td>92485682</td>
    </tr>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.74630</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.41810</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.81970</td>
      <td>72822507</td>
    </tr>
    <tr>
      <th>14</th>
      <td>SNP_15</td>
      <td>BovineHD1000000224</td>
      <td>10</td>
      <td>1.78774</td>
      <td>814291</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7df1db76-4200-403f-a46d-c42bd76f9e15')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7df1db76-4200-403f-a46d-c42bd76f9e15 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7df1db76-4200-403f-a46d-c42bd76f9e15');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### SNP chrom별로 분류해 데이터 살펴보기


```python
plt.figure(figsize=(10,6))
sns.countplot(x='chrom', data=snp_info)
plt.title('chrom 종류별 개수')
```




    Text(0.5, 1.0, 'chrom 종류별 개수')




    
![png](output_19_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_info)
plt.title('유전거리 분포')
```




    Text(0.5, 1.0, '유전거리 분포')




    
![png](output_20_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_info)
plt.title('각 마커 유전체상 위치정보')
```




    Text(0.5, 1.0, '각 마커 유전체상 위치정보')




    
![png](output_21_1.png)
    


#### SNP chrom=6인 경우 살펴보기
chrom=6인 종류의 유전체 정보가 가장 많기 때문에


```python
snp_6=snp_info[snp_info['chrom']==6]
snp_6
```





  <div id="df-ef40a192-b4ff-4960-8c1b-69b2adb99a33">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>SNP_02</td>
      <td>ARS-USMARC-Parent-DQ647190-rs29013632</td>
      <td>6</td>
      <td>31.1567</td>
      <td>13897068</td>
    </tr>
    <tr>
      <th>2</th>
      <td>SNP_03</td>
      <td>ARS-BFGL-NGS-117009</td>
      <td>6</td>
      <td>68.2892</td>
      <td>44649549</td>
    </tr>
    <tr>
      <th>3</th>
      <td>SNP_04</td>
      <td>ARS-BFGL-NGS-60567</td>
      <td>6</td>
      <td>77.8749</td>
      <td>53826064</td>
    </tr>
    <tr>
      <th>4</th>
      <td>SNP_05</td>
      <td>BovineHD0600017032</td>
      <td>6</td>
      <td>80.5015</td>
      <td>61779512</td>
    </tr>
    <tr>
      <th>5</th>
      <td>SNP_06</td>
      <td>BovineHD0600017424</td>
      <td>6</td>
      <td>80.5954</td>
      <td>63048481</td>
    </tr>
    <tr>
      <th>6</th>
      <td>SNP_07</td>
      <td>Hapmap49442-BTA-111073</td>
      <td>6</td>
      <td>80.7800</td>
      <td>64037334</td>
    </tr>
    <tr>
      <th>7</th>
      <td>SNP_08</td>
      <td>BovineHD0600018638</td>
      <td>6</td>
      <td>82.6856</td>
      <td>67510588</td>
    </tr>
    <tr>
      <th>8</th>
      <td>SNP_09</td>
      <td>ARS-BFGL-NGS-37727</td>
      <td>6</td>
      <td>86.8740</td>
      <td>73092782</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ef40a192-b4ff-4960-8c1b-69b2adb99a33')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ef40a192-b4ff-4960-8c1b-69b2adb99a33 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ef40a192-b4ff-4960-8c1b-69b2adb99a33');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




#### SNP chrom=9인 경우 살펴보기
다음으로 많은 chrom=9인 종류의 유전체 정보


```python
snp_9=snp_info[snp_info['chrom']==9]
snp_9
```





  <div id="df-5179591b-d935-41fb-bc68-ae2ccff7ea6d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SNP_id</th>
      <th>name</th>
      <th>chrom</th>
      <th>cm</th>
      <th>pos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>11</th>
      <td>SNP_12</td>
      <td>Hapmap32827-BTA-146530</td>
      <td>9</td>
      <td>62.7463</td>
      <td>55007839</td>
    </tr>
    <tr>
      <th>12</th>
      <td>SNP_13</td>
      <td>BTB-00395482</td>
      <td>9</td>
      <td>63.4181</td>
      <td>59692848</td>
    </tr>
    <tr>
      <th>13</th>
      <td>SNP_14</td>
      <td>Hapmap40256-BTA-84189</td>
      <td>9</td>
      <td>66.8197</td>
      <td>72822507</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5179591b-d935-41fb-bc68-ae2ccff7ea6d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-5179591b-d935-41fb-bc68-ae2ccff7ea6d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5179591b-d935-41fb-bc68-ae2ccff7ea6d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_6)
plt.title('chrom=6 인 SNP의 유전거리')
```




    Text(0.5, 1.0, 'chrom=6 인 SNP의 유전거리')




    
![png](output_26_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_6)
plt.title('chrom=6 인 SNP의 유전체상 위치정보')
```




    Text(0.5, 1.0, 'chrom=6 인 SNP의 유전체상 위치정보')




    
![png](output_27_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='cm', data=snp_9)
plt.title('chrom=9 인 SNP의 유전거리')
```




    Text(0.5, 1.0, 'chrom=9 인 SNP의 유전거리')




    
![png](output_28_1.png)
    



```python
plt.figure(figsize=(10,6))
sns.barplot(x='SNP_id', y='pos', data=snp_9)
plt.title('chrom=9 인 SNP의 유전체상 위치정보')
```




    Text(0.5, 1.0, 'chrom=9 인 SNP의 유전체상 위치정보')




    
![png](output_29_1.png)
    


chrom=6인 데이터로 판단할 가능성이 높다고 생각해 나머지 데이터들은 모두 제거할 계획

## 문자형 데이터 -> 숫자형으로 변경


```python
def get_x_y(df):
    if 'class' in df.columns:
        df_x = df.drop(columns=['id', 'class'])
        df_y = df['class']
        return df_x, df_y
    else:
        df_x = df.drop(columns=['id'])
        return df_x
```


```python
train_x, train_y = get_x_y(train)
test_x = get_x_y(test)
```


```python
class_le = preprocessing.LabelEncoder()
snp_le = preprocessing.LabelEncoder()
snp_col = [f'SNP_{str(x).zfill(2)}' for x in range(1,16)]
```


```python
snp_data = []
for col in snp_col:
    snp_data += list(train_x[col].values)
```


```python
train_y = class_le.fit_transform(train_y)
snp_le.fit(snp_data)
```




    LabelEncoder()




```python
for col in train_x.columns:
    if col in snp_col:
        train_x[col] = snp_le.transform(train_x[col])
        test_x[col] = snp_le.transform(test_x[col])
```


```python
train_x
```





  <div id="df-7e755782-49fb-46e9-be67-9994417822be">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>father</th>
      <th>mother</th>
      <th>gender</th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>261</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 19 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7e755782-49fb-46e9-be67-9994417822be')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7e755782-49fb-46e9-be67-9994417822be button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7e755782-49fb-46e9-be67-9994417822be');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
snp_le.classes_
```




    array(['A A', 'A G', 'C A', 'C C', 'G A', 'G G'], dtype='<U3')




```python
trait_1=train_x[train_x['trait']==1]
trait_2=train_x[train_x['trait']==2]
```


```python
for i in trait_1.columns[4:]:
    print(trait_1[i].unique())
```

    [0 1 5]
    [5 1]
    [0]
    [4 5 0]
    [0 2]
    [5 1]
    [5 4]
    [0 4]
    [5 4 0]
    [1 5 0]
    [5 1]
    [5 4 0]
    [5 1]
    [0 3 2]
    [5 4 0]



```python
for i in trait_2.columns[4:]:
    print(trait_2[i].unique())
```

    [5 1 0]
    [1 5 0]
    [0 2 3]
    [4 0 5]
    [2 0 3]
    [0 1 5]
    [0 4]
    [5 4 0]
    [0 4 5]
    [5 1 0]
    [1 0 5]
    [0 4 5]
    [0 5 1]
    [0 2 3]
    [0 4 5]


trait 종류에 따른 데이터 차이 없음 파악

하지만 표현형 정보이기 때문에 제외하지 않는 것이 더 좋다는 판단


```python
train_num=train_x.drop(columns=['father', 'mother', 'gender'])
train_num
```





  <div id="df-6dd66c5a-8de5-4286-92d7-aaee2cb04982">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trait</th>
      <th>SNP_01</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
      <th>SNP_10</th>
      <th>SNP_11</th>
      <th>SNP_12</th>
      <th>SNP_13</th>
      <th>SNP_14</th>
      <th>SNP_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>5</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 16 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-6dd66c5a-8de5-4286-92d7-aaee2cb04982')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-6dd66c5a-8de5-4286-92d7-aaee2cb04982 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-6dd66c5a-8de5-4286-92d7-aaee2cb04982');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
colormap = plt.cm.PuBu
plt.figure(figsize=(15,15), dpi=200)

sns.heatmap(train_num.astype(float).corr(), linewidths = 0.1, vmax = 1.0,
           square = True, cmap = colormap, linecolor = "white", annot = True, annot_kws = {"size" : 5})
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fe0746f2460>




    
![png](output_45_1.png)
    


## 모델 돌려보기


```python
model_y=pd.DataFrame(train_y, columns=['class'])
model_y
```





  <div id="df-7d64a0cb-b7db-4b53-9f60-7ea9079d18da">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>1</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
    </tr>
    <tr>
      <th>259</th>
      <td>0</td>
    </tr>
    <tr>
      <th>260</th>
      <td>0</td>
    </tr>
    <tr>
      <th>261</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7d64a0cb-b7db-4b53-9f60-7ea9079d18da')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7d64a0cb-b7db-4b53-9f60-7ea9079d18da button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7d64a0cb-b7db-4b53-9f60-7ea9079d18da');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
model_X=train_x.drop(columns=['father','mother','gender', 'SNP_01','SNP_10', 'SNP_11','SNP_12','SNP_13','SNP_14', 'SNP_15' ])
model_X
```





  <div id="df-cdaa6184-9cd4-44e8-bba9-3c8511bbb551">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trait</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>257</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>258</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>259</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>260</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>261</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>262 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-cdaa6184-9cd4-44e8-bba9-3c8511bbb551')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-cdaa6184-9cd4-44e8-bba9-3c8511bbb551 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-cdaa6184-9cd4-44e8-bba9-3c8511bbb551');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




chrom=6이 아닌 경우 제외하고 모두 제거


```python
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
```


```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5, shuffle=True)

models = []
accuracy = []
for train_idx, valid_idx in kfold.split(model_X, model_y):
  X_train, X_valid = model_X.iloc[train_idx], model_X.iloc[valid_idx]
  y_train, y_valid = model_y.iloc[train_idx], model_y.iloc[valid_idx]

  # display(X_valid)
  # model = ExtraTreesClassifier(random_state=0) # Gooooood!!
  # model = RandomForestClassifier(random_state=0) # Good
  model = XGBClassifier(random_state=0) # Good
  

  model.fit(X_train, y_train)

  y_pred = model.predict(X_valid)
  models.append(model)
  accuracy.append(accuracy_score(y_valid, y_pred))
```


```python
accuracy
```




    [0.9245283018867925,
     0.9622641509433962,
     0.9230769230769231,
     0.9038461538461539,
     0.9807692307692307]



분류 모델로 xgboost를 사용해 정확도 확인해보았습니다.

## 제출해보기


```python
test_x=test_x.drop(columns=['father','mother','gender', 'SNP_01','SNP_10', 'SNP_11','SNP_12','SNP_13','SNP_14', 'SNP_15'])
```


```python
test_x
```





  <div id="df-70d25ec5-d2f1-4aad-a345-77c42dae7059">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>trait</th>
      <th>SNP_02</th>
      <th>SNP_03</th>
      <th>SNP_04</th>
      <th>SNP_05</th>
      <th>SNP_06</th>
      <th>SNP_07</th>
      <th>SNP_08</th>
      <th>SNP_09</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>171</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>172</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>173</th>
      <td>2</td>
      <td>5</td>
      <td>2</td>
      <td>4</td>
      <td>3</td>
      <td>5</td>
      <td>0</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>174</th>
      <td>2</td>
      <td>5</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-70d25ec5-d2f1-4aad-a345-77c42dae7059')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-70d25ec5-d2f1-4aad-a345-77c42dae7059 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-70d25ec5-d2f1-4aad-a345-77c42dae7059');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
y_pred=[]
```


```python
y_pred = model.predict(test_x)
```


```python
y_pred
```




    array([0, 1, 2, 2, 0, 2, 2, 1, 0, 0, 2, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
           1, 0, 1, 2, 1, 0, 0, 2, 0, 0, 1, 1, 0, 2, 2, 1, 1, 2, 0, 1, 2, 1,
           1, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 1, 1, 0, 1, 2, 0, 1, 2, 2, 2, 0,
           1, 0, 0, 1, 1, 1, 0, 0, 2, 1, 2, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1,
           2, 0, 1, 1, 2, 1, 1, 2, 0, 1, 0, 2, 0, 1, 1, 1, 0, 0, 2, 1, 0, 1,
           2, 1, 1, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 0, 1, 0, 0,
           1, 1, 1, 2, 0, 0, 1, 0, 0, 0, 2, 1, 1, 1, 0, 1, 2, 0, 0, 1, 0, 1,
           1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 0, 2, 1, 1, 0, 1, 2, 2, 1, 1])




```python
submit = pd.read_csv('/content/drive/MyDrive/유전체/sample_submission.csv')

submit['class'] = class_le.inverse_transform(y_pred)

submit.to_csv('/content/drive/MyDrive/유전체/sample_submission_blog.csv', index=False)
```


```python
submit
```





  <div id="df-7e7b74d4-e73c-4a69-935a-97f9e8f80d4d">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TEST_000</td>
      <td>A</td>
    </tr>
    <tr>
      <th>1</th>
      <td>TEST_001</td>
      <td>B</td>
    </tr>
    <tr>
      <th>2</th>
      <td>TEST_002</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TEST_003</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>TEST_004</td>
      <td>A</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>170</th>
      <td>TEST_170</td>
      <td>B</td>
    </tr>
    <tr>
      <th>171</th>
      <td>TEST_171</td>
      <td>C</td>
    </tr>
    <tr>
      <th>172</th>
      <td>TEST_172</td>
      <td>C</td>
    </tr>
    <tr>
      <th>173</th>
      <td>TEST_173</td>
      <td>B</td>
    </tr>
    <tr>
      <th>174</th>
      <td>TEST_174</td>
      <td>B</td>
    </tr>
  </tbody>
</table>
<p>175 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7e7b74d4-e73c-4a69-935a-97f9e8f80d4d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-7e7b74d4-e73c-4a69-935a-97f9e8f80d4d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7e7b74d4-e73c-4a69-935a-97f9e8f80d4d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




- 결과

리더보드 상 96%의 정확도를 예측했습니다.

전처리를 조금 더 연구하고 어떤 모델이 더 적합한지 확인해보고 다시 제출할 예정입니다.
