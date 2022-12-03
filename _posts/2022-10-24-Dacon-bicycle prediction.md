---
layout: single
title: "Dacon - bicycle prediction"
---

데이콘 따릉이 예측 코드입니다.




```python
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv]
!rm ~/.cache/matplotlib -rf
```

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    fonts-nanum is already the newest version (20170925-1).
    The following package was automatically installed and is no longer required:
      libnvidia-common-460
    Use 'sudo apt autoremove' to remove it.
    0 upgraded, 0 newly installed, 0 to remove and 49 not upgraded.
    fc-cache: invalid option -- ']'
    usage: fc-cache [-EfrsvVh] [-y SYSROOT] [--error-on-no-fonts] [--force|--really-force] [--sysroot=SYSROOT] [--system-only] [--verbose] [--version] [--help] [dirs]
    Build font information caches in [dirs]
    (all directories in font configuration by default).
    
      -E, --error-on-no-fonts  raise an error if no fonts in a directory
      -f, --force              scan directories with apparently valid caches
      -r, --really-force       erase all existing caches, then rescan
      -s, --system-only        scan system-wide directories only
      -y, --sysroot=SYSROOT    prepend SYSROOT to all paths for scanning
      -v, --verbose            display status information while busy
      -V, --version            display font config version and exit
      -h, --help               display this help and exit



```python
# 드라이브 사용 설정
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


# 1. 데이터 살펴보기

train.csv : 모델을 학습하는 데 사용하는 데이터  
test.csv : 모델을 통하여 따릉이 대여량을 예측하는 데 사용하는 데이터  
sample_submission.csv : test데이터의 예측 값을 저장하여 제출하기 위한 데이터

## 1.1. 데이터 준비

read_csv() 메소드 이용하기


```python
# 데이터 로드
import pandas as pd

train = pd.read_csv('/content/drive/MyDrive/dacon.io/suwon/5회차/data/train.csv')
test = pd.read_csv('/content/drive/MyDrive/dacon.io/suwon/5회차/data/test.csv')
submission = pd.read_csv('/content/drive/MyDrive/dacon.io/suwon/5회차/data/sample_submission.csv')
```

## 1.2. 데이터 확인

head()  
tail()  
info()  
shape


```python
# 데이터 상위 5개 행 출력
train.head()
```





  <div id="df-c47ad70c-fe69-4859-9400-e68963fe25bc">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-01</td>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-02</td>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-03</td>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-04</td>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-05</td>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c47ad70c-fe69-4859-9400-e68963fe25bc')"
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
          document.querySelector('#df-c47ad70c-fe69-4859-9400-e68963fe25bc button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c47ad70c-fe69-4859-9400-e68963fe25bc');
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
# 데이터 하위 5개 행 출력
train.tail()
```





  <div id="df-858e26ea-37f4-4d6c-bd51-30758314fb95">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>268</th>
      <td>2020-06-26</td>
      <td>228.662</td>
      <td>3.980</td>
      <td>0.223</td>
      <td>2.271</td>
      <td>78.378</td>
      <td>20.500</td>
      <td>27.526</td>
      <td>36.486</td>
      <td>96150</td>
    </tr>
    <tr>
      <th>269</th>
      <td>2020-06-27</td>
      <td>207.770</td>
      <td>2.865</td>
      <td>0.081</td>
      <td>1.794</td>
      <td>78.412</td>
      <td>20.812</td>
      <td>28.842</td>
      <td>21.081</td>
      <td>107001</td>
    </tr>
    <tr>
      <th>270</th>
      <td>2020-06-28</td>
      <td>282.568</td>
      <td>1.730</td>
      <td>0.000</td>
      <td>1.820</td>
      <td>72.736</td>
      <td>21.000</td>
      <td>29.053</td>
      <td>7.297</td>
      <td>98568</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2020-06-29</td>
      <td>137.027</td>
      <td>2.257</td>
      <td>0.088</td>
      <td>2.043</td>
      <td>70.473</td>
      <td>19.625</td>
      <td>26.000</td>
      <td>15.541</td>
      <td>70053</td>
    </tr>
    <tr>
      <th>272</th>
      <td>2020-06-30</td>
      <td>120.797</td>
      <td>3.622</td>
      <td>0.432</td>
      <td>5.574</td>
      <td>77.061</td>
      <td>19.125</td>
      <td>26.053</td>
      <td>41.284</td>
      <td>38086</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-858e26ea-37f4-4d6c-bd51-30758314fb95')"
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
          document.querySelector('#df-858e26ea-37f4-4d6c-bd51-30758314fb95 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-858e26ea-37f4-4d6c-bd51-30758314fb95');
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
# 데이터 결측치 및 변수들의 타입 확인
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 273 entries, 0 to 272
    Data columns (total 10 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   date_time                  273 non-null    object 
     1   wind_direction             273 non-null    float64
     2   sky_condition              273 non-null    float64
     3   precipitation_form         273 non-null    float64
     4   wind_speed                 273 non-null    float64
     5   humidity                   273 non-null    float64
     6   low_temp                   273 non-null    float64
     7   high_temp                  273 non-null    float64
     8   Precipitation_Probability  273 non-null    float64
     9   number_of_rentals          273 non-null    int64  
    dtypes: float64(8), int64(1), object(1)
    memory usage: 21.5+ KB



```python
# 데이터 행, 열 출력
train.shape
```




    (273, 10)



## 1.3. 데이터 통계치 확인


```python
train.describe()
```





  <div id="df-188a1a44-11fe-4a2d-9363-e2fbf6916a10">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
      <td>273.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>202.750967</td>
      <td>2.288256</td>
      <td>0.100963</td>
      <td>2.480963</td>
      <td>56.745491</td>
      <td>13.795249</td>
      <td>23.384733</td>
      <td>16.878103</td>
      <td>59574.978022</td>
    </tr>
    <tr>
      <th>std</th>
      <td>56.659232</td>
      <td>0.961775</td>
      <td>0.203193</td>
      <td>0.884397</td>
      <td>12.351268</td>
      <td>5.107711</td>
      <td>5.204605</td>
      <td>16.643772</td>
      <td>27659.575774</td>
    </tr>
    <tr>
      <th>min</th>
      <td>57.047000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.753000</td>
      <td>24.831000</td>
      <td>1.938000</td>
      <td>9.895000</td>
      <td>0.000000</td>
      <td>1037.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>171.541000</td>
      <td>1.405000</td>
      <td>0.000000</td>
      <td>1.820000</td>
      <td>47.196000</td>
      <td>9.938000</td>
      <td>19.842000</td>
      <td>4.054000</td>
      <td>36761.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>209.774000</td>
      <td>2.167000</td>
      <td>0.000000</td>
      <td>2.411000</td>
      <td>55.845000</td>
      <td>14.375000</td>
      <td>24.158000</td>
      <td>12.162000</td>
      <td>63032.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>238.412000</td>
      <td>3.000000</td>
      <td>0.088000</td>
      <td>2.924000</td>
      <td>66.419000</td>
      <td>18.000000</td>
      <td>27.526000</td>
      <td>22.973000</td>
      <td>81515.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>321.622000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>5.607000</td>
      <td>88.885000</td>
      <td>22.312000</td>
      <td>33.421000</td>
      <td>82.162000</td>
      <td>110377.000000</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-188a1a44-11fe-4a2d-9363-e2fbf6916a10')"
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
          document.querySelector('#df-188a1a44-11fe-4a2d-9363-e2fbf6916a10 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-188a1a44-11fe-4a2d-9363-e2fbf6916a10');
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




# 2. 데이터 전처리

EDA를 하기 위한 데이터 선행처리

## 2.1. 'date_time' 열 쪼개기


```python
train['date_time']
```




    0      2018-04-01
    1      2018-04-02
    2      2018-04-03
    3      2018-04-04
    4      2018-04-05
              ...    
    268    2020-06-26
    269    2020-06-27
    270    2020-06-28
    271    2020-06-29
    272    2020-06-30
    Name: date_time, Length: 273, dtype: object




```python
train['date_time'][0].split('-')
```




    ['2018', '04', '01']




```python
year, month, day = train['date_time'][0].split('-')
print('년 : ' + year)
print('월 : ' + month)
print('일 : ' + day)
```

    년 : 2018
    월 : 04
    일 : 01



```python
train['date_time'].apply(lambda x : x.split('-'))
```




    0      [2018, 04, 01]
    1      [2018, 04, 02]
    2      [2018, 04, 03]
    3      [2018, 04, 04]
    4      [2018, 04, 05]
                ...      
    268    [2020, 06, 26]
    269    [2020, 06, 27]
    270    [2020, 06, 28]
    271    [2020, 06, 29]
    272    [2020, 06, 30]
    Name: date_time, Length: 273, dtype: object




```python
train['date_time'].apply(lambda x : x.split('-')[0])
```




    0      2018
    1      2018
    2      2018
    3      2018
    4      2018
           ... 
    268    2020
    269    2020
    270    2020
    271    2020
    272    2020
    Name: date_time, Length: 273, dtype: object




```python
def temp(x):
  return x.split('-')[0]

train['date_time'].map(temp)
```




    0      2018
    1      2018
    2      2018
    3      2018
    4      2018
           ... 
    268    2020
    269    2020
    270    2020
    271    2020
    272    2020
    Name: date_time, Length: 273, dtype: object




```python
train.apply(lambda x : x*2)
```





  <div id="df-ce54dcc6-53f6-4da7-bc0c-b45277b000ab">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-012018-04-01</td>
      <td>415.000</td>
      <td>8.000</td>
      <td>0.000</td>
      <td>6.100</td>
      <td>150.000</td>
      <td>25.200</td>
      <td>42.000</td>
      <td>60.000</td>
      <td>45988</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-022018-04-02</td>
      <td>416.634</td>
      <td>5.900</td>
      <td>0.000</td>
      <td>6.556</td>
      <td>139.666</td>
      <td>25.624</td>
      <td>38.000</td>
      <td>39.000</td>
      <td>56278</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-032018-04-03</td>
      <td>427.032</td>
      <td>5.822</td>
      <td>0.000</td>
      <td>5.380</td>
      <td>149.758</td>
      <td>20.624</td>
      <td>30.632</td>
      <td>38.226</td>
      <td>53634</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-042018-04-04</td>
      <td>287.672</td>
      <td>7.384</td>
      <td>0.850</td>
      <td>6.276</td>
      <td>143.698</td>
      <td>16.624</td>
      <td>24.736</td>
      <td>86.986</td>
      <td>52068</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-052018-04-05</td>
      <td>191.810</td>
      <td>8.000</td>
      <td>1.446</td>
      <td>6.372</td>
      <td>147.568</td>
      <td>11.750</td>
      <td>20.842</td>
      <td>126.756</td>
      <td>5666</td>
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
    </tr>
    <tr>
      <th>268</th>
      <td>2020-06-262020-06-26</td>
      <td>457.324</td>
      <td>7.960</td>
      <td>0.446</td>
      <td>4.542</td>
      <td>156.756</td>
      <td>41.000</td>
      <td>55.052</td>
      <td>72.972</td>
      <td>192300</td>
    </tr>
    <tr>
      <th>269</th>
      <td>2020-06-272020-06-27</td>
      <td>415.540</td>
      <td>5.730</td>
      <td>0.162</td>
      <td>3.588</td>
      <td>156.824</td>
      <td>41.624</td>
      <td>57.684</td>
      <td>42.162</td>
      <td>214002</td>
    </tr>
    <tr>
      <th>270</th>
      <td>2020-06-282020-06-28</td>
      <td>565.136</td>
      <td>3.460</td>
      <td>0.000</td>
      <td>3.640</td>
      <td>145.472</td>
      <td>42.000</td>
      <td>58.106</td>
      <td>14.594</td>
      <td>197136</td>
    </tr>
    <tr>
      <th>271</th>
      <td>2020-06-292020-06-29</td>
      <td>274.054</td>
      <td>4.514</td>
      <td>0.176</td>
      <td>4.086</td>
      <td>140.946</td>
      <td>39.250</td>
      <td>52.000</td>
      <td>31.082</td>
      <td>140106</td>
    </tr>
    <tr>
      <th>272</th>
      <td>2020-06-302020-06-30</td>
      <td>241.594</td>
      <td>7.244</td>
      <td>0.864</td>
      <td>11.148</td>
      <td>154.122</td>
      <td>38.250</td>
      <td>52.106</td>
      <td>82.568</td>
      <td>76172</td>
    </tr>
  </tbody>
</table>
<p>273 rows × 10 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ce54dcc6-53f6-4da7-bc0c-b45277b000ab')"
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
          document.querySelector('#df-ce54dcc6-53f6-4da7-bc0c-b45277b000ab button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ce54dcc6-53f6-4da7-bc0c-b45277b000ab');
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
train['date_time'].map(lambda x : x*2)
```




    0      2018-04-012018-04-01
    1      2018-04-022018-04-02
    2      2018-04-032018-04-03
    3      2018-04-042018-04-04
    4      2018-04-052018-04-05
                   ...         
    268    2020-06-262020-06-26
    269    2020-06-272020-06-27
    270    2020-06-282020-06-28
    271    2020-06-292020-06-29
    272    2020-06-302020-06-30
    Name: date_time, Length: 273, dtype: object




```python
train['year'] = train['date_time'].apply(lambda x : x.split('-')[0]) #년
train['month'] = train['date_time'].apply(lambda x : x.split('-')[1]) #월
train['day'] = train['date_time'].apply(lambda x : x.split('-')[2]) #일

# 혹은 
#train['year'] = train['date_time'].map(lambda x : x.split('-')[0]) #년
#train['month'] = train['date_time'].map(lambda x : x.split('-')[1]) #월
#train['day'] = train['date_time'].map(lambda x : x.split('-')[2]) #일
```


```python
train.head()
```





  <div id="df-a379a8fe-4d4f-46ab-8eb5-73ce73c0018a">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-01</td>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
      <td>2018</td>
      <td>04</td>
      <td>01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-02</td>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
      <td>2018</td>
      <td>04</td>
      <td>02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-03</td>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
      <td>2018</td>
      <td>04</td>
      <td>03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-04</td>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
      <td>2018</td>
      <td>04</td>
      <td>04</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-05</td>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
      <td>2018</td>
      <td>04</td>
      <td>05</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a379a8fe-4d4f-46ab-8eb5-73ce73c0018a')"
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
          document.querySelector('#df-a379a8fe-4d4f-46ab-8eb5-73ce73c0018a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a379a8fe-4d4f-46ab-8eb5-73ce73c0018a');
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




## 2.2. 요일 정보 추가 (week_day 열 생성)


```python
week_day = pd.to_datetime(train['date_time']).dt.day_name()
week_day
```




    0         Sunday
    1         Monday
    2        Tuesday
    3      Wednesday
    4       Thursday
             ...    
    268       Friday
    269     Saturday
    270       Sunday
    271       Monday
    272      Tuesday
    Name: date_time, Length: 273, dtype: object




```python
train['week_day']
```




    0      0
    1      1
    2      2
    3      3
    4      4
          ..
    268    5
    269    6
    270    0
    271    1
    272    2
    Name: week_day, Length: 273, dtype: int64




```python
train['week_day'] = week_day

train.head()
```





  <div id="df-5ab87a48-1077-476d-9fd4-13d34ef54af5">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-04-01</td>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
      <td>2018</td>
      <td>04</td>
      <td>01</td>
      <td>Sunday</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-04-02</td>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
      <td>2018</td>
      <td>04</td>
      <td>02</td>
      <td>Monday</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-04-03</td>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
      <td>2018</td>
      <td>04</td>
      <td>03</td>
      <td>Tuesday</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-04-04</td>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
      <td>2018</td>
      <td>04</td>
      <td>04</td>
      <td>Wednesday</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-04-05</td>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
      <td>2018</td>
      <td>04</td>
      <td>05</td>
      <td>Thursday</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-5ab87a48-1077-476d-9fd4-13d34ef54af5')"
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
          document.querySelector('#df-5ab87a48-1077-476d-9fd4-13d34ef54af5 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-5ab87a48-1077-476d-9fd4-13d34ef54af5');
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




## 2.3. 가공한 변수 타입 변경


```python
train['year'] = train['year'].apply(lambda x : int(x))
train['year'] = train['year'].astype('int')
```


```python
train['month'] = train['month'].astype('int')
train['day'] = train['day'].astype('int')
```


```python
train.week_day.unique()
```




    array(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
           'Saturday'], dtype=object)




```python
train.loc[train['week_day'] == 'Sunday', 'week_day'] = 0
train.loc[train['week_day'] == 'Monday', 'week_day'] = 1
train.loc[train['week_day'] == 'Tuesday', 'week_day'] = 2
train.loc[train['week_day'] == 'Wednesday', 'week_day'] = 3
train.loc[train['week_day'] == 'Thursday', 'week_day'] = 4
train.loc[train['week_day'] == 'Friday', 'week_day'] = 5
train.loc[train['week_day'] == 'Saturday', 'week_day'] = 6
```


```python
train['week_day'] = train['week_day'].astype('int')
```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 273 entries, 0 to 272
    Data columns (total 14 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   date_time                  273 non-null    object 
     1   wind_direction             273 non-null    float64
     2   sky_condition              273 non-null    float64
     3   precipitation_form         273 non-null    float64
     4   wind_speed                 273 non-null    float64
     5   humidity                   273 non-null    float64
     6   low_temp                   273 non-null    float64
     7   high_temp                  273 non-null    float64
     8   Precipitation_Probability  273 non-null    float64
     9   number_of_rentals          273 non-null    int64  
     10  year                       273 non-null    int64  
     11  month                      273 non-null    int64  
     12  day                        273 non-null    int64  
     13  week_day                   273 non-null    int64  
    dtypes: float64(8), int64(5), object(1)
    memory usage: 30.0+ KB


# 3. EDA

가설 설정

1. 주중에 비해서 주말에 따릉이 대여량이 증가 할 것이다.
2. 날씨가 덥고 습하다면 따릉이 대여량이 감소 할 것이다.
3. 날씨가 춥고 바람이 많이 분다면 따릉이 대여량이 감소 할 것이다.

## 3.1. 시각화


```python
import matplotlib.pyplot as plt
import seaborn as sns 

%matplotlib inline
plt.rcParams['font.size'] = 15
plt.style.use('ggplot')
plt.rc('font', family='NanumGothic')
```

### 3.1.1. Target


```python
plt.figure(dpi = 150)

x = train['number_of_rentals'].index
y = train['number_of_rentals'].values

plt.title("따릉이 대여량")
plt.xlabel("일별")
plt.ylabel("일별 따릉이 대여량")
plt.hlines(y=8000, xmin=0, xmax=len(x), color='Orange', linestyle='dotted')
plt.plot(x,y)
plt.show()
```


    
![output_39_0](https://user-images.githubusercontent.com/113446739/197523373-dc79b338-9388-48f6-a7ba-5486cfe69d3f.png)

    


* 인사이트

1. 시간이 흐를수록 대여량이 증가하고있다 -> 연도별 차이
2. 이상치가 존재한다
3. 평일에는 출근을 하니까 대여량이 주말보다 많을까? -> 요일별 차이

## 3.1.2. 연도별 따릉이 대여량

인사이트1 : 시간이 흐를수록 대여량이 증가하고있다 


```python
month_day = train['month'].astype(str) + '_' + train['day'].astype(str)

plt.figure(figsize=(10,5))
sns.scatterplot(x=month_day, y=train['number_of_rentals'], hue=train['year'])
plt.xticks(rotation=45)
plt.title('연도별 따릉이 대여량 비교', fontsize = 30)
plt.show()
```


    
![output_42_0](https://user-images.githubusercontent.com/113446739/197523910-040517c9-007c-4b3c-b88e-b3097c84e560.png)
    


* 결론

Year은 매우 중요한 변수이고,

Year을 반영하는 파생 변수를 생성해도 좋겠다!

## 3.1.3.  평일과 주말 대여량

인사이트 3 : 평일에는 출근을 하니까 대여량이 주말보다 많을까?

가설 1


```python
weekend = train[(train['week_day'] == 0) | (train['week_day'] == 6)] # 일요일과 토요일 -> 주말을 나타내는 데이터 프레임
```


```python
weekday = train[(train['week_day'] != 0) & (train['week_day'] != 6)] # 일요일과 토요일 아닌 -> 평일을 나타내는 데이터 프레임
```


```python
weekend.mean().number_of_rentals
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      """Entry point for launching an IPython kernel.





    58988.64102564102




```python
weekday.mean().number_of_rentals
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      """Entry point for launching an IPython kernel.





    59809.51282051282




```python
x = ['weekday', 'weekend']
y = [weekday.mean().number_of_rentals, weekend.mean().number_of_rentals]

plt.figure(figsize = (8,5))
plt.title("따릉이 대여량 평균")
plt.xlabel('평일/주말')
plt.ylabel('대여량')
plt.bar(x, y)
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: Dropping of nuisance columns in DataFrame reductions (with 'numeric_only=None') is deprecated; in a future version this will raise TypeError.  Select only valid columns before calling the reduction.
      



    
![output_49_1](https://user-images.githubusercontent.com/113446739/197523992-b7a9ff4f-6f6e-4823-9729-07ba703c43df.png)
    



```python
weekday.number_of_rentals.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f438b032690>




    
![output_50_1](https://user-images.githubusercontent.com/113446739/197524063-51066037-dd6c-4345-aa42-e1a127d9baa5.png)
    



```python
weekend.number_of_rentals.plot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f438b3411d0>




    
![output_51_1](https://user-images.githubusercontent.com/113446739/197524161-e91bb6f1-c9c5-4b64-be84-11b9f511290f.png)
    


* 결론

평일과 주말은 별 차이가 없다고 말할 수 있다!

## 3.1.4. 요일별 대여량


```python
plt.figure(figsize=(15,8))

sns.barplot(x=train['week_day'], y=train['number_of_rentals'])
plt.title('요일별 대여량 비교', fontsize = 30)
plt.show()
```


    
![output_54_0](https://user-images.githubusercontent.com/113446739/197524220-edda4b17-eb57-4a8b-9e06-3f7ba86a18d6.png)    



```python
plt.figure(figsize=(15,8))

sns.barplot(x=train['week_day'], y=train['number_of_rentals'], hue=train['year'])
plt.title('요일별+연도별 대여량 비교', fontsize = 30)
plt.show()
```


    
![output_55_0](https://user-images.githubusercontent.com/113446739/197524271-6c78077b-d7af-473d-ae80-fe751ee2fa70.png)
    


* 결론

요일이라는 변수는 따릉이 대여량에 영향을 크게 미치지 않는다

따라서 drop을 해줘도 괜찮겠다!

## 3.1.5 더위+습도 = 더위 점수

가설2. 날씨가 덥고 습하다면 따릉이 대여량 감소


```python
hot_score = train['high_temp'] * train['humidity']

plt.figure(figsize=(8,5))
sns.scatterplot(x=hot_score, y=train['number_of_rentals'], hue=train['year'], s = 100 )
plt.xticks(fontsize=15)
plt.title('더위 점수와 대여량', fontsize = 30)
plt.show()
```


    
![output_58_0](https://user-images.githubusercontent.com/113446739/197524326-aa1dc950-aeb1-47a3-98ca-9ac11facfb8c.png)
    


* 결론

더위 점수가 따릉이 대여량이 영향을 준다  
: 파생변수 '더위 점수'

## 3.1.6. 추위 점수

가설3 : 날씨가 춥고 바람이 많이 불면 따릉이 대여량이 감소할 것이다.


```python
cold_score = train['wind_speed'] / train['low_temp']

plt.figure(figsize=(8,5))
sns.scatterplot(x = cold_score, y=train['number_of_rentals'], hue=train['year'], s = 50 )
plt.xticks(fontsize=15)
plt.title('추위 점수와 대여량', fontsize = 30)
plt.show()
```


    
![output_61_0](https://user-images.githubusercontent.com/113446739/197524407-69136ae5-25a8-48d6-a216-b61e7eba890a.png)
    


* 결론 

추위 점수가 따릉이 대여량에 영향을 미칠 것이다 (음의 상관관계)

: 추위 점수라는 파생변수를 생성

## 3.1.7. 이상치 확인


```python
plt.style.use("ggplot")

feature = train.describe().columns

plt.figure(figsize=(30,15))
plt.suptitle("독립변수별 이상치 확인", fontsize = 30)

for i in range(len(feature)):
  plt.subplot(2,7,i+1)
  plt.title(feature[i])
  plt.boxplot(train[feature[i]])
plt.show()
```


    
![output_64_0](https://user-images.githubusercontent.com/113446739/197524474-9cb75a82-6207-4361-a165-7d7aa486ee8b.png)    


## 3.1.8. 상관관계


```python
plt.figure(figsize=(12,10))
sns.heatmap(data = train.corr(method='pearson'), annot=True, fmt='.2f', linewidths=.5, cmap='Blues')
```

    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:214: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0.0, flags=flags)
    /usr/local/lib/python3.7/dist-packages/matplotlib/backends/backend_agg.py:183: RuntimeWarning: Glyph 8722 missing from current font.
      font.set_text(s, 0, flags=flags)





    <matplotlib.axes._subplots.AxesSubplot at 0x7f438a7249d0>




    
![output_66_2](https://user-images.githubusercontent.com/113446739/197524541-637c321c-0ee3-4502-aa5b-5892415bc1e7.png)
    


## 3.2. 가설 검정 및 총 정리

3.1.1. : 타겟을 통하여 3가지 인사이트 도출  
3.1.2. : 연도별 대여량이 증가 하므로 , Year 관한 변수 하나 생성  
3.1.3. : 가설 1 : 평일과 주말의 대여량 차이는 X  
3.1.4. : 가설 1 (파생) : 요일별 대여량도 차이가 없다  
3.1.5. : 가설 2 : 더위 점수는 대여량이 감소하는 것이아니라 오히려 증가했다  
3.1.6. : 가설 3 : 추위 점수는 대여량이 감소하였다.  
3.1.7. : 이상치에서는 특이한 이상치 발견 X  
3.1.8. : 피어슨 상관계수를 통한 전반적인 피처의 상관관계를 도출  

: Year 인플레이션 변수 생성, 더위 점수 생성, 추위 점수 생성, 요일, 날짜, 일 변수 삭제

# 4. Feature Engineering

## 4.1. Feature Extraction

### 4.1.1. Year inflation


```python
train[train['year']==2018]['number_of_rentals']
```




    0     22994
    1     28139
    2     26817
    3     26034
    4      2833
          ...  
    86     6391
    87    41128
    88    27757
    89    42765
    90    26911
    Name: number_of_rentals, Length: 91, dtype: int64




```python
train[train['year']==2019]['number_of_rentals']
```




    91     37660
    92     42029
    93     43257
    94     45681
    95     45643
           ...  
    177    81274
    178    79234
    179    77056
    180    67346
    181    74994
    Name: number_of_rentals, Length: 91, dtype: int64




```python
train[train['year']==2020]['number_of_rentals']
```




    182     70258
    183     72129
    184     74856
    185     66405
    186     64111
            ...  
    268     96150
    269    107001
    270     98568
    271     70053
    272     38086
    Name: number_of_rentals, Length: 91, dtype: int64




```python
x1 = sum(train[train['year']==2018]['number_of_rentals'])

x2 = sum(train[train['year']==2019]['number_of_rentals'])

x3 = sum(train[train['year']==2020]['number_of_rentals'])
```


```python
x1
```




    2860617




```python
x2
```




    5994774




```python
x3
```




    7408578




```python
print(x3/x1)
print(x3/x2)
```

    2.5898531680403214
    1.2358394161314505



```python
y1 = train[train['year']==2018]['number_of_rentals'] * 2.5898531680403214
y2 = train[train['year']==2019]['number_of_rentals'] * 1.2358394161314505
y3 = train[train['year']==2020]['number_of_rentals']

pd.concat([y1,y2,y3], axis=0).to_frame()
```





  <div id="df-67518690-bab1-4809-883f-54962b4df01f">
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
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>59551.083746</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72875.878295</td>
    </tr>
    <tr>
      <th>2</th>
      <td>69452.092407</td>
    </tr>
    <tr>
      <th>3</th>
      <td>67424.237377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7337.054025</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>268</th>
      <td>96150.000000</td>
    </tr>
    <tr>
      <th>269</th>
      <td>107001.000000</td>
    </tr>
    <tr>
      <th>270</th>
      <td>98568.000000</td>
    </tr>
    <tr>
      <th>271</th>
      <td>70053.000000</td>
    </tr>
    <tr>
      <th>272</th>
      <td>38086.000000</td>
    </tr>
  </tbody>
</table>
<p>273 rows × 1 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-67518690-bab1-4809-883f-54962b4df01f')"
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
          document.querySelector('#df-67518690-bab1-4809-883f-54962b4df01f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-67518690-bab1-4809-883f-54962b4df01f');
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
temp = train[['year', 'number_of_rentals']]
temp['inflation_rentals'] = pd.concat([y1,y2,y3], axis=0).to_frame()
temp[:183]
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      






  <div id="df-3869f203-caa2-41af-b3ec-01ac8e973e02">
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
      <th>year</th>
      <th>number_of_rentals</th>
      <th>inflation_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018</td>
      <td>22994</td>
      <td>59551.083746</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>28139</td>
      <td>72875.878295</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018</td>
      <td>26817</td>
      <td>69452.092407</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018</td>
      <td>26034</td>
      <td>67424.237377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018</td>
      <td>2833</td>
      <td>7337.054025</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>178</th>
      <td>2019</td>
      <td>79234</td>
      <td>97920.500298</td>
    </tr>
    <tr>
      <th>179</th>
      <td>2019</td>
      <td>77056</td>
      <td>95228.842049</td>
    </tr>
    <tr>
      <th>180</th>
      <td>2019</td>
      <td>67346</td>
      <td>83228.841319</td>
    </tr>
    <tr>
      <th>181</th>
      <td>2019</td>
      <td>74994</td>
      <td>92680.541173</td>
    </tr>
    <tr>
      <th>182</th>
      <td>2020</td>
      <td>70258</td>
      <td>70258.000000</td>
    </tr>
  </tbody>
</table>
<p>183 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-3869f203-caa2-41af-b3ec-01ac8e973e02')"
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
          document.querySelector('#df-3869f203-caa2-41af-b3ec-01ac8e973e02 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-3869f203-caa2-41af-b3ec-01ac8e973e02');
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





  <div id="df-dbe046c0-bb80-455e-9a22-535411dca2c2">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-04-01</td>
      <td>108.833</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>2.900</td>
      <td>28.333</td>
      <td>11.800</td>
      <td>20.667</td>
      <td>18.333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-04-02</td>
      <td>116.717</td>
      <td>3.850</td>
      <td>0.000</td>
      <td>2.662</td>
      <td>46.417</td>
      <td>12.000</td>
      <td>19.000</td>
      <td>28.500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-04-03</td>
      <td>82.669</td>
      <td>4.000</td>
      <td>0.565</td>
      <td>2.165</td>
      <td>77.258</td>
      <td>8.875</td>
      <td>16.368</td>
      <td>52.847</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-04-04</td>
      <td>44.123</td>
      <td>3.466</td>
      <td>0.466</td>
      <td>3.747</td>
      <td>63.288</td>
      <td>6.250</td>
      <td>17.368</td>
      <td>37.671</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-04-05</td>
      <td>147.791</td>
      <td>1.500</td>
      <td>0.000</td>
      <td>1.560</td>
      <td>48.176</td>
      <td>7.188</td>
      <td>18.684</td>
      <td>4.459</td>
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
      <th>86</th>
      <td>2021-06-26</td>
      <td>177.149</td>
      <td>3.980</td>
      <td>0.223</td>
      <td>1.066</td>
      <td>74.628</td>
      <td>20.312</td>
      <td>28.579</td>
      <td>36.486</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2021-06-27</td>
      <td>138.723</td>
      <td>2.777</td>
      <td>0.135</td>
      <td>1.290</td>
      <td>70.236</td>
      <td>20.812</td>
      <td>29.000</td>
      <td>18.378</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2021-06-28</td>
      <td>111.095</td>
      <td>3.338</td>
      <td>1.270</td>
      <td>1.692</td>
      <td>70.338</td>
      <td>21.000</td>
      <td>28.789</td>
      <td>35.946</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2021-06-29</td>
      <td>171.622</td>
      <td>3.270</td>
      <td>0.595</td>
      <td>1.470</td>
      <td>70.473</td>
      <td>21.000</td>
      <td>29.421</td>
      <td>27.770</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2021-06-30</td>
      <td>181.709</td>
      <td>3.270</td>
      <td>0.703</td>
      <td>1.180</td>
      <td>75.203</td>
      <td>21.500</td>
      <td>30.211</td>
      <td>29.054</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 9 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dbe046c0-bb80-455e-9a22-535411dca2c2')"
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
          document.querySelector('#df-dbe046c0-bb80-455e-9a22-535411dca2c2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dbe046c0-bb80-455e-9a22-535411dca2c2');
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




### 4.1.2. 피쳐 제거


```python
train = train.drop(['week_day'], axis = 1)
train = train.drop(['day'], axis = 1)
train = train.drop(['date_time'], axis = 1)
```


```python
train.head()
```





  <div id="df-ab12a49a-6784-4557-bfb4-93914cc5e5c0">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>year</th>
      <th>month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
      <td>2018</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
      <td>2018</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
      <td>2018</td>
      <td>4</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
      <td>2018</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
      <td>2018</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ab12a49a-6784-4557-bfb4-93914cc5e5c0')"
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
          document.querySelector('#df-ab12a49a-6784-4557-bfb4-93914cc5e5c0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ab12a49a-6784-4557-bfb4-93914cc5e5c0');
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




### 4.1.3. 더위 및 추위 파생 변수 생성


```python
train['hot_score'] = train['high_temp']*train['humidity']
train['cold_score'] = train['wind_speed']/train['low_temp']
```


```python
train.head()
```





  <div id="df-d1f381db-e6bf-494f-9b56-86db3daf1a0f">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
      <td>2018</td>
      <td>4</td>
      <td>1575.000000</td>
      <td>0.242063</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
      <td>2018</td>
      <td>4</td>
      <td>1326.827000</td>
      <td>0.255854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
      <td>2018</td>
      <td>4</td>
      <td>1146.846764</td>
      <td>0.260861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
      <td>2018</td>
      <td>4</td>
      <td>888.628432</td>
      <td>0.377526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
      <td>2018</td>
      <td>4</td>
      <td>768.903064</td>
      <td>0.542298</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-d1f381db-e6bf-494f-9b56-86db3daf1a0f')"
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
          document.querySelector('#df-d1f381db-e6bf-494f-9b56-86db3daf1a0f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-d1f381db-e6bf-494f-9b56-86db3daf1a0f');
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




## 4.2. Feature Selection


```python
#X = train.drop(['number_of_rentals'], axis = 1)
X = train[['wind_direction', 'sky_condition', 'precipitation_form', 'wind_speed',
       'humidity', 'low_temp', 'high_temp', 'Precipitation_Probability',
       'year', 'month', 'hot_score', 'cold_score']]

y = train['number_of_rentals']
```


```python
train.columns
```




    Index(['wind_direction', 'sky_condition', 'precipitation_form', 'wind_speed',
           'humidity', 'low_temp', 'high_temp', 'Precipitation_Probability',
           'number_of_rentals', 'year', 'month', 'hot_score', 'cold_score'],
          dtype='object')



# 5. Modeling

## 5.1. 모델 선택


```python
#from sklearn.ensemble import RandomForestRegressor

#model = RandomForestRegressor()
```


```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

## 5.2. 모델 학습


```python
model.fit(X,y)
```




    LinearRegression()



## 5.3. 모델 예측

### 5.3.1. test 셋 데이터 처리


```python
X
```





  <div id="df-48871732-0313-49d6-b3b2-f65cc302e05b">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>2018</td>
      <td>4</td>
      <td>1575.000000</td>
      <td>0.242063</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>2018</td>
      <td>4</td>
      <td>1326.827000</td>
      <td>0.255854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>2018</td>
      <td>4</td>
      <td>1146.846764</td>
      <td>0.260861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>2018</td>
      <td>4</td>
      <td>888.628432</td>
      <td>0.377526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2018</td>
      <td>4</td>
      <td>768.903064</td>
      <td>0.542298</td>
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
    </tr>
    <tr>
      <th>268</th>
      <td>228.662</td>
      <td>3.980</td>
      <td>0.223</td>
      <td>2.271</td>
      <td>78.378</td>
      <td>20.500</td>
      <td>27.526</td>
      <td>36.486</td>
      <td>2020</td>
      <td>6</td>
      <td>2157.432828</td>
      <td>0.110780</td>
    </tr>
    <tr>
      <th>269</th>
      <td>207.770</td>
      <td>2.865</td>
      <td>0.081</td>
      <td>1.794</td>
      <td>78.412</td>
      <td>20.812</td>
      <td>28.842</td>
      <td>21.081</td>
      <td>2020</td>
      <td>6</td>
      <td>2261.558904</td>
      <td>0.086200</td>
    </tr>
    <tr>
      <th>270</th>
      <td>282.568</td>
      <td>1.730</td>
      <td>0.000</td>
      <td>1.820</td>
      <td>72.736</td>
      <td>21.000</td>
      <td>29.053</td>
      <td>7.297</td>
      <td>2020</td>
      <td>6</td>
      <td>2113.199008</td>
      <td>0.086667</td>
    </tr>
    <tr>
      <th>271</th>
      <td>137.027</td>
      <td>2.257</td>
      <td>0.088</td>
      <td>2.043</td>
      <td>70.473</td>
      <td>19.625</td>
      <td>26.000</td>
      <td>15.541</td>
      <td>2020</td>
      <td>6</td>
      <td>1832.298000</td>
      <td>0.104102</td>
    </tr>
    <tr>
      <th>272</th>
      <td>120.797</td>
      <td>3.622</td>
      <td>0.432</td>
      <td>5.574</td>
      <td>77.061</td>
      <td>19.125</td>
      <td>26.053</td>
      <td>41.284</td>
      <td>2020</td>
      <td>6</td>
      <td>2007.670233</td>
      <td>0.291451</td>
    </tr>
  </tbody>
</table>
<p>273 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-48871732-0313-49d6-b3b2-f65cc302e05b')"
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
          document.querySelector('#df-48871732-0313-49d6-b3b2-f65cc302e05b button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-48871732-0313-49d6-b3b2-f65cc302e05b');
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




### 5.3.1.1.  데이터 전처리 과정에서


```python
# date_time 쪼개기 -> year, month, day
test['year'] = test['date_time'].apply(lambda x : x.split('-')[0])
test['month'] = test['date_time'].apply(lambda x : x.split('-')[1])
test['day'] = test['date_time'].apply(lambda x : x.split('-')[2])

# 요일 정보 추가
week_day = pd.to_datetime(test['date_time']).dt.day_name()
test['week_day'] = week_day

# 가공한 변수 타입 변경
test['year'] = test['year'].astype('int')
test['month'] = test['month'].astype('int')
test['day'] = test['day'].astype('int')

test.loc[test['week_day'] == 'Sunday', 'week_day'] = 0
test.loc[test['week_day'] == 'Monday', 'week_day'] = 1
test.loc[test['week_day'] == 'Tuesday', 'week_day'] = 2
test.loc[test['week_day'] == 'Wednesday', 'week_day'] = 3
test.loc[test['week_day'] == 'Thursday', 'week_day'] = 4
test.loc[test['week_day'] == 'Friday', 'week_day'] = 5
test.loc[test['week_day'] == 'Saturday', 'week_day'] = 6
test['week_day'] = test['week_day'].astype('int')

test.head()
```





  <div id="df-c68407fa-f395-4227-aad0-14d86fe971bb">
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
      <th>date_time</th>
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>week_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-04-01</td>
      <td>108.833</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>2.900</td>
      <td>28.333</td>
      <td>11.800</td>
      <td>20.667</td>
      <td>18.333</td>
      <td>2021</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-04-02</td>
      <td>116.717</td>
      <td>3.850</td>
      <td>0.000</td>
      <td>2.662</td>
      <td>46.417</td>
      <td>12.000</td>
      <td>19.000</td>
      <td>28.500</td>
      <td>2021</td>
      <td>4</td>
      <td>2</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-04-03</td>
      <td>82.669</td>
      <td>4.000</td>
      <td>0.565</td>
      <td>2.165</td>
      <td>77.258</td>
      <td>8.875</td>
      <td>16.368</td>
      <td>52.847</td>
      <td>2021</td>
      <td>4</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-04-04</td>
      <td>44.123</td>
      <td>3.466</td>
      <td>0.466</td>
      <td>3.747</td>
      <td>63.288</td>
      <td>6.250</td>
      <td>17.368</td>
      <td>37.671</td>
      <td>2021</td>
      <td>4</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-04-05</td>
      <td>147.791</td>
      <td>1.500</td>
      <td>0.000</td>
      <td>1.560</td>
      <td>48.176</td>
      <td>7.188</td>
      <td>18.684</td>
      <td>4.459</td>
      <td>2021</td>
      <td>4</td>
      <td>5</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c68407fa-f395-4227-aad0-14d86fe971bb')"
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
          document.querySelector('#df-c68407fa-f395-4227-aad0-14d86fe971bb button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c68407fa-f395-4227-aad0-14d86fe971bb');
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




#### 5.4.1.2. Feature Extraction 과정


```python
# 피쳐 제거
test = test.drop(['week_day'],axis = 1)
test = test.drop(['day'],axis = 1)
test = test.drop(['date_time'],axis = 1)

# 더위 점수
test['hot_score'] = test['high_temp'] * test['humidity']

# 추위 점수
test['cold_score'] = test['wind_speed'] / test['low_temp']

test.head()
```





  <div id="df-7df9d8fb-20f2-4748-9afd-e0c0b954a091">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108.833</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>2.900</td>
      <td>28.333</td>
      <td>11.800</td>
      <td>20.667</td>
      <td>18.333</td>
      <td>2021</td>
      <td>4</td>
      <td>585.558111</td>
      <td>0.245763</td>
    </tr>
    <tr>
      <th>1</th>
      <td>116.717</td>
      <td>3.850</td>
      <td>0.000</td>
      <td>2.662</td>
      <td>46.417</td>
      <td>12.000</td>
      <td>19.000</td>
      <td>28.500</td>
      <td>2021</td>
      <td>4</td>
      <td>881.923000</td>
      <td>0.221833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>82.669</td>
      <td>4.000</td>
      <td>0.565</td>
      <td>2.165</td>
      <td>77.258</td>
      <td>8.875</td>
      <td>16.368</td>
      <td>52.847</td>
      <td>2021</td>
      <td>4</td>
      <td>1264.558944</td>
      <td>0.243944</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.123</td>
      <td>3.466</td>
      <td>0.466</td>
      <td>3.747</td>
      <td>63.288</td>
      <td>6.250</td>
      <td>17.368</td>
      <td>37.671</td>
      <td>2021</td>
      <td>4</td>
      <td>1099.185984</td>
      <td>0.599520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>147.791</td>
      <td>1.500</td>
      <td>0.000</td>
      <td>1.560</td>
      <td>48.176</td>
      <td>7.188</td>
      <td>18.684</td>
      <td>4.459</td>
      <td>2021</td>
      <td>4</td>
      <td>900.120384</td>
      <td>0.217028</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-7df9d8fb-20f2-4748-9afd-e0c0b954a091')"
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
          document.querySelector('#df-7df9d8fb-20f2-4748-9afd-e0c0b954a091 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-7df9d8fb-20f2-4748-9afd-e0c0b954a091');
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
X
```





  <div id="df-144e42e8-3248-4116-b1ac-aadb42aabcd0">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>2018</td>
      <td>4</td>
      <td>1575.000000</td>
      <td>0.242063</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>2018</td>
      <td>4</td>
      <td>1326.827000</td>
      <td>0.255854</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>2018</td>
      <td>4</td>
      <td>1146.846764</td>
      <td>0.260861</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>2018</td>
      <td>4</td>
      <td>888.628432</td>
      <td>0.377526</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2018</td>
      <td>4</td>
      <td>768.903064</td>
      <td>0.542298</td>
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
    </tr>
    <tr>
      <th>268</th>
      <td>228.662</td>
      <td>3.980</td>
      <td>0.223</td>
      <td>2.271</td>
      <td>78.378</td>
      <td>20.500</td>
      <td>27.526</td>
      <td>36.486</td>
      <td>2020</td>
      <td>6</td>
      <td>2157.432828</td>
      <td>0.110780</td>
    </tr>
    <tr>
      <th>269</th>
      <td>207.770</td>
      <td>2.865</td>
      <td>0.081</td>
      <td>1.794</td>
      <td>78.412</td>
      <td>20.812</td>
      <td>28.842</td>
      <td>21.081</td>
      <td>2020</td>
      <td>6</td>
      <td>2261.558904</td>
      <td>0.086200</td>
    </tr>
    <tr>
      <th>270</th>
      <td>282.568</td>
      <td>1.730</td>
      <td>0.000</td>
      <td>1.820</td>
      <td>72.736</td>
      <td>21.000</td>
      <td>29.053</td>
      <td>7.297</td>
      <td>2020</td>
      <td>6</td>
      <td>2113.199008</td>
      <td>0.086667</td>
    </tr>
    <tr>
      <th>271</th>
      <td>137.027</td>
      <td>2.257</td>
      <td>0.088</td>
      <td>2.043</td>
      <td>70.473</td>
      <td>19.625</td>
      <td>26.000</td>
      <td>15.541</td>
      <td>2020</td>
      <td>6</td>
      <td>1832.298000</td>
      <td>0.104102</td>
    </tr>
    <tr>
      <th>272</th>
      <td>120.797</td>
      <td>3.622</td>
      <td>0.432</td>
      <td>5.574</td>
      <td>77.061</td>
      <td>19.125</td>
      <td>26.053</td>
      <td>41.284</td>
      <td>2020</td>
      <td>6</td>
      <td>2007.670233</td>
      <td>0.291451</td>
    </tr>
  </tbody>
</table>
<p>273 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-144e42e8-3248-4116-b1ac-aadb42aabcd0')"
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
          document.querySelector('#df-144e42e8-3248-4116-b1ac-aadb42aabcd0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-144e42e8-3248-4116-b1ac-aadb42aabcd0');
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





  <div id="df-ddd8b7a3-79f1-4e70-8c94-166b83de7334">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>108.833</td>
      <td>3.000</td>
      <td>0.000</td>
      <td>2.900</td>
      <td>28.333</td>
      <td>11.800</td>
      <td>20.667</td>
      <td>18.333</td>
      <td>2021</td>
      <td>4</td>
      <td>585.558111</td>
      <td>0.245763</td>
    </tr>
    <tr>
      <th>1</th>
      <td>116.717</td>
      <td>3.850</td>
      <td>0.000</td>
      <td>2.662</td>
      <td>46.417</td>
      <td>12.000</td>
      <td>19.000</td>
      <td>28.500</td>
      <td>2021</td>
      <td>4</td>
      <td>881.923000</td>
      <td>0.221833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>82.669</td>
      <td>4.000</td>
      <td>0.565</td>
      <td>2.165</td>
      <td>77.258</td>
      <td>8.875</td>
      <td>16.368</td>
      <td>52.847</td>
      <td>2021</td>
      <td>4</td>
      <td>1264.558944</td>
      <td>0.243944</td>
    </tr>
    <tr>
      <th>3</th>
      <td>44.123</td>
      <td>3.466</td>
      <td>0.466</td>
      <td>3.747</td>
      <td>63.288</td>
      <td>6.250</td>
      <td>17.368</td>
      <td>37.671</td>
      <td>2021</td>
      <td>4</td>
      <td>1099.185984</td>
      <td>0.599520</td>
    </tr>
    <tr>
      <th>4</th>
      <td>147.791</td>
      <td>1.500</td>
      <td>0.000</td>
      <td>1.560</td>
      <td>48.176</td>
      <td>7.188</td>
      <td>18.684</td>
      <td>4.459</td>
      <td>2021</td>
      <td>4</td>
      <td>900.120384</td>
      <td>0.217028</td>
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
    </tr>
    <tr>
      <th>86</th>
      <td>177.149</td>
      <td>3.980</td>
      <td>0.223</td>
      <td>1.066</td>
      <td>74.628</td>
      <td>20.312</td>
      <td>28.579</td>
      <td>36.486</td>
      <td>2021</td>
      <td>6</td>
      <td>2132.793612</td>
      <td>0.052481</td>
    </tr>
    <tr>
      <th>87</th>
      <td>138.723</td>
      <td>2.777</td>
      <td>0.135</td>
      <td>1.290</td>
      <td>70.236</td>
      <td>20.812</td>
      <td>29.000</td>
      <td>18.378</td>
      <td>2021</td>
      <td>6</td>
      <td>2036.844000</td>
      <td>0.061983</td>
    </tr>
    <tr>
      <th>88</th>
      <td>111.095</td>
      <td>3.338</td>
      <td>1.270</td>
      <td>1.692</td>
      <td>70.338</td>
      <td>21.000</td>
      <td>28.789</td>
      <td>35.946</td>
      <td>2021</td>
      <td>6</td>
      <td>2024.960682</td>
      <td>0.080571</td>
    </tr>
    <tr>
      <th>89</th>
      <td>171.622</td>
      <td>3.270</td>
      <td>0.595</td>
      <td>1.470</td>
      <td>70.473</td>
      <td>21.000</td>
      <td>29.421</td>
      <td>27.770</td>
      <td>2021</td>
      <td>6</td>
      <td>2073.386133</td>
      <td>0.070000</td>
    </tr>
    <tr>
      <th>90</th>
      <td>181.709</td>
      <td>3.270</td>
      <td>0.703</td>
      <td>1.180</td>
      <td>75.203</td>
      <td>21.500</td>
      <td>30.211</td>
      <td>29.054</td>
      <td>2021</td>
      <td>6</td>
      <td>2271.957833</td>
      <td>0.054884</td>
    </tr>
  </tbody>
</table>
<p>91 rows × 12 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ddd8b7a3-79f1-4e70-8c94-166b83de7334')"
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
          document.querySelector('#df-ddd8b7a3-79f1-4e70-8c94-166b83de7334 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ddd8b7a3-79f1-4e70-8c94-166b83de7334');
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




### 5.3.2. test셋 예측


```python
predict = model.predict(test)
```


```python
predict
```




    array([ 96175.64422127,  93845.2098333 ,  77136.79275706,  76208.34011316,
           104855.47606267, 106151.69058622, 108028.43207671, 104456.69087511,
           107049.61077241, 105968.83659898, 103553.04714487,  79431.8352027 ,
            85630.55931145, 102031.38304323, 104799.14644321,  83444.92445652,
            88860.34030706, 103701.39082906, 110096.04867059, 114773.7680036 ,
           115376.65527379, 107053.24553972,  99015.85620129, 106986.65589715,
           105795.71112359, 104720.48530408, 102351.88821785,  96771.34554236,
           100850.12918711,  88416.91561056,  78834.49708716, 109977.13681276,
           112504.84864921,  89176.17025888, 100837.20247494, 109083.76655378,
            93503.81284279, 103481.52202641, 109211.69545331, 105960.66052474,
           116639.39492839, 115846.06947344, 122330.21946926, 119195.13290808,
            99595.97522175,  81599.03243487,  97948.02504173, 118050.20750616,
           120096.93855339,  97598.98280108,  94825.61985469, 116749.34810233,
           115334.62314846, 113371.90227321, 100559.56877531, 108029.56615018,
           100249.66067819,  94639.13848754, 111815.25823662, 111824.40902916,
            99469.90342809, 121443.50873075, 121037.47718177,  99788.51299746,
           118085.52985217, 116575.40874914, 120770.12690849, 120769.14763585,
           120437.08719463, 125490.38895093, 117335.81213796,  92162.36222453,
           127267.81403804, 128191.46953645, 125862.6798288 , 111233.70925906,
           112634.71597277, 117169.85179579, 103984.22794942, 124053.68943377,
           120667.17799323, 124930.43024421, 103234.08558206,  86482.28691185,
           111516.62785649, 121460.53870281, 113243.00833053, 118748.50985421,
            91910.98951595, 109340.97605734, 108098.18301832])




```python
submission['number_of_rentals'] = predict
submission.head()
```





  <div id="df-22db85bb-e1a5-42d8-9dc6-77b6fc36fbd6">
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
      <th>date_time</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-04-01</td>
      <td>96175.644221</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-04-02</td>
      <td>93845.209833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-04-03</td>
      <td>77136.792757</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-04-04</td>
      <td>76208.340113</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-04-05</td>
      <td>104855.476063</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-22db85bb-e1a5-42d8-9dc6-77b6fc36fbd6')"
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
          document.querySelector('#df-22db85bb-e1a5-42d8-9dc6-77b6fc36fbd6 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-22db85bb-e1a5-42d8-9dc6-77b6fc36fbd6');
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
submission.to_csv('submission_3.csv', index = False)
```

## Feature Engineering : Year Inflaction : 2020 -> 2021


```python
x1 = sum(train[train['year']==2018]['number_of_rentals'])

x2 = sum(train[train['year']==2019]['number_of_rentals'])

x3 = sum(train[train['year']==2020]['number_of_rentals'])

x4 = sum(submission['number_of_rentals']) ## 2021
```


```python
x1
```




    2860617




```python
x4
```




    9685804.67883733




```python
print(x4/x1)
print(x4/x2)
print(x4/x3)
```

    3.3859145348144577
    1.6157080615278123
    1.3073770268514862



```python
# lr
def rental_rate_change(df):
    y1 = df[df['year'] == 2018]['number_of_rentals'] * 3.3859145348144577
    y2 = df[df['year'] == 2019]['number_of_rentals'] * 1.6157080615278123
    y3 = df[df['year'] == 2020]['number_of_rentals'] * 1.3073770268514862
    new = pd.concat([y1, y2, y3], axis=0).to_frame()
    df['inflaction_rentals'] = new['number_of_rentals']
    return True
```


```python
rental_rate_change(train)
```




    True




```python
train.head()
```





  <div id="df-36edf37d-4c84-4b58-bb46-b21e51047624">
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
      <th>wind_direction</th>
      <th>sky_condition</th>
      <th>precipitation_form</th>
      <th>wind_speed</th>
      <th>humidity</th>
      <th>low_temp</th>
      <th>high_temp</th>
      <th>Precipitation_Probability</th>
      <th>number_of_rentals</th>
      <th>year</th>
      <th>month</th>
      <th>hot_score</th>
      <th>cold_score</th>
      <th>inflaction_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>207.500</td>
      <td>4.000</td>
      <td>0.000</td>
      <td>3.050</td>
      <td>75.000</td>
      <td>12.600</td>
      <td>21.000</td>
      <td>30.000</td>
      <td>22994</td>
      <td>2018</td>
      <td>4</td>
      <td>1575.000000</td>
      <td>0.242063</td>
      <td>77855.718814</td>
    </tr>
    <tr>
      <th>1</th>
      <td>208.317</td>
      <td>2.950</td>
      <td>0.000</td>
      <td>3.278</td>
      <td>69.833</td>
      <td>12.812</td>
      <td>19.000</td>
      <td>19.500</td>
      <td>28139</td>
      <td>2018</td>
      <td>4</td>
      <td>1326.827000</td>
      <td>0.255854</td>
      <td>95276.249095</td>
    </tr>
    <tr>
      <th>2</th>
      <td>213.516</td>
      <td>2.911</td>
      <td>0.000</td>
      <td>2.690</td>
      <td>74.879</td>
      <td>10.312</td>
      <td>15.316</td>
      <td>19.113</td>
      <td>26817</td>
      <td>2018</td>
      <td>4</td>
      <td>1146.846764</td>
      <td>0.260861</td>
      <td>90800.070080</td>
    </tr>
    <tr>
      <th>3</th>
      <td>143.836</td>
      <td>3.692</td>
      <td>0.425</td>
      <td>3.138</td>
      <td>71.849</td>
      <td>8.312</td>
      <td>12.368</td>
      <td>43.493</td>
      <td>26034</td>
      <td>2018</td>
      <td>4</td>
      <td>888.628432</td>
      <td>0.377526</td>
      <td>88148.898999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>95.905</td>
      <td>4.000</td>
      <td>0.723</td>
      <td>3.186</td>
      <td>73.784</td>
      <td>5.875</td>
      <td>10.421</td>
      <td>63.378</td>
      <td>2833</td>
      <td>2018</td>
      <td>4</td>
      <td>768.903064</td>
      <td>0.542298</td>
      <td>9592.295877</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-36edf37d-4c84-4b58-bb46-b21e51047624')"
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
          document.querySelector('#df-36edf37d-4c84-4b58-bb46-b21e51047624 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-36edf37d-4c84-4b58-bb46-b21e51047624');
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
X = train.drop(['number_of_rentals', 'inflaction_rentals'], axis=1 )
y = train['inflaction_rentals']
```


```python
model = RandomForestRegressor()
```


```python
model.fit(X,y)
```




    RandomForestRegressor()




```python
predict = model.predict(test)
```


```python
submission['number_of_rentals'] = predict
submission.head()
```





  <div id="df-a60bd4e4-3a7b-42c9-89b4-9af093ee0fa0">
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
      <th>date_time</th>
      <th>number_of_rentals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-04-01</td>
      <td>100112.727581</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-04-02</td>
      <td>95841.116392</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-04-03</td>
      <td>42917.868837</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-04-04</td>
      <td>50312.338813</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-04-05</td>
      <td>98285.098696</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-a60bd4e4-3a7b-42c9-89b4-9af093ee0fa0')"
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
          document.querySelector('#df-a60bd4e4-3a7b-42c9-89b4-9af093ee0fa0 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-a60bd4e4-3a7b-42c9-89b4-9af093ee0fa0');
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
submission.to_csv('submission_4.csv', index = False)
```


```python

```
