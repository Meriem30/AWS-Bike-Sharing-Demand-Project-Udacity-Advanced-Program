# Predict Bike Sharing Demand with AutoGluon Template

## Project: Predict Bike Sharing Demand with AutoGluon
This notebook is a template with each step that you need to complete for the project.

Please fill in your code where there are explicit `?` markers in the notebook. You are welcome to add more cells and code as you see fit.

Once you have completed all the code implementations, please export your notebook as a HTML file so the reviews can view your code. Make sure you have all outputs correctly outputted.

`File-> Export Notebook As... -> Export Notebook as HTML`

There is a writeup to complete as well after all code implememtation is done. Please answer all questions and attach the necessary tables and charts. You can complete the writeup in either markdown or PDF.

Completing the code template and writeup template will cover all of the rubric points for this project.

The rubric contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this notebook and also discuss the results in the writeup file.

## Step 1: Create an account with Kaggle

### Create Kaggle Account and download API key
Below is example of steps to get the API username and key. Each student will have their own username and key.

1. Open account settings.
![kaggle1.png](kaggle1.png)
![kaggle2.png](kaggle2.png)
2. Scroll down to API and click Create New API Token.
![kaggle3.png](kaggle3.png)
![kaggle4.png](kaggle4.png)
3. Open up `kaggle.json` and use the username and key.
![kaggle5.png](kaggle5.png)

## Step 2: Download the Kaggle dataset using the kaggle python library

### Open up Sagemaker Studio and use starter template

1. Notebook should be using a `ml.t3.medium` instance (2 vCPU + 4 GiB)
2. Notebook should be using kernal: `Python 3 (MXNet 1.8 Python 3.7 CPU Optimized)`

### Install packages


```python
!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh==2.0.1
!pip install autogluon --no-cache-dir
# Without --no-cache-dir, smaller aws instances may have trouble installing
```

    Requirement already satisfied: pip in /opt/conda/lib/python3.12/site-packages (25.1.1)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.12/site-packages (80.9.0)
    Requirement already satisfied: wheel in /opt/conda/lib/python3.12/site-packages (0.45.1)
    Collecting mxnet<2.0.0
      Using cached mxnet-1.9.1-py3-none-manylinux2014_x86_64.whl.metadata (3.4 kB)
    Collecting bokeh==2.0.1
      Using cached bokeh-2.0.1.tar.gz (8.6 MB)
      Preparing metadata (setup.py) ... [?25lerror
      [1;31merror[0m: [1msubprocess-exited-with-error[0m
      
      [31m×[0m [32mpython setup.py egg_info[0m did not run successfully.
      [31m│[0m exit code: [1;36m1[0m
      [31m╰─>[0m [31m[21 lines of output][0m
      [31m   [0m /tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/versioneer.py:416: SyntaxWarning: invalid escape sequence '\s'
      [31m   [0m   LONG_VERSION_PY['git'] = '''
      [31m   [0m Traceback (most recent call last):
      [31m   [0m   File "<string>", line 2, in <module>
      [31m   [0m   File "<pip-setuptools-caller>", line 35, in <module>
      [31m   [0m   File "/tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/setup.py", line 118, in <module>
      [31m   [0m     version=get_version(),
      [31m   [0m             ^^^^^^^^^^^^^
      [31m   [0m   File "/tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/_setup_support.py", line 243, in get_version
      [31m   [0m     return versioneer.get_version()
      [31m   [0m            ^^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m   File "/tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/versioneer.py", line 1484, in get_version
      [31m   [0m     return get_versions()["version"]
      [31m   [0m            ^^^^^^^^^^^^^^
      [31m   [0m   File "/tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/versioneer.py", line 1416, in get_versions
      [31m   [0m     cfg = get_config_from_root(root)
      [31m   [0m           ^^^^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m   File "/tmp/pip-install-x6_bo_ej/bokeh_6911c04ac8ae4c46af0d0111deeaecea/versioneer.py", line 340, in get_config_from_root
      [31m   [0m     parser = configparser.SafeConfigParser()
      [31m   [0m              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      [31m   [0m AttributeError: module 'configparser' has no attribute 'SafeConfigParser'. Did you mean: 'RawConfigParser'?
      [31m   [0m [31m[end of output][0m
      
      [1;35mnote[0m: This error originates from a subprocess, and is likely not a problem with pip.
    [1;31merror[0m: [1mmetadata-generation-failed[0m
    
    [31m×[0m Encountered error while generating package metadata.
    [31m╰─>[0m See above for output.
    
    [1;35mnote[0m: This is an issue with the package mentioned above, not pip.
    [1;36mhint[0m: See above for details.
    [?25hRequirement already satisfied: autogluon in /opt/conda/lib/python3.12/site-packages (1.3.0)
    Requirement already satisfied: autogluon.core==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core[all]==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: autogluon.features==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon) (1.3.0)
    Requirement already satisfied: autogluon.tabular==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: autogluon.multimodal==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon) (1.3.0)
    Requirement already satisfied: autogluon.timeseries==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries[all]==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: numpy<2.3.0,>=1.25.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.26.4)
    Requirement already satisfied: scipy<1.16,>=1.5.4 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.15.2)
    Requirement already satisfied: scikit-learn<1.7.0,>=1.4.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.6.1)
    Requirement already satisfied: networkx<4,>=3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.4.2)
    Requirement already satisfied: pandas<2.3.0,>=2.0.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2.2.3)
    Requirement already satisfied: tqdm<5,>=4.38 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (4.67.1)
    Requirement already satisfied: requests in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2.32.3)
    Requirement already satisfied: matplotlib<3.11,>=3.7.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.10.3)
    Requirement already satisfied: boto3<2,>=1.10 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.37.1)
    Requirement already satisfied: autogluon.common==1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: psutil<7.1.0,>=5.7.3 in /opt/conda/lib/python3.12/site-packages (from autogluon.common==1.3.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (5.9.8)
    Requirement already satisfied: hyperopt<0.2.8,>=0.2.7 in /opt/conda/lib/python3.12/site-packages (from autogluon.core[all]==1.3.0->autogluon) (0.2.7)
    Requirement already satisfied: ray<2.45,>=2.10.0 in /opt/conda/lib/python3.12/site-packages (from ray[default,tune]<2.45,>=2.10.0; extra == "all"->autogluon.core[all]==1.3.0->autogluon) (2.44.1)
    Requirement already satisfied: pyarrow>=15.0.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.core[all]==1.3.0->autogluon) (19.0.1)
    Requirement already satisfied: Pillow<12,>=10.0.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (11.2.1)
    Requirement already satisfied: torch<2.7,>=2.2 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (2.6.0)
    Requirement already satisfied: lightning<2.7,>=2.2 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (2.5.1.post0)
    Requirement already satisfied: transformers<4.50,>=4.38.0 in /opt/conda/lib/python3.12/site-packages (from transformers[sentencepiece]<4.50,>=4.38.0->autogluon.multimodal==1.3.0->autogluon) (4.49.0)
    Requirement already satisfied: accelerate<2.0,>=0.34.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.5.2)
    Requirement already satisfied: jsonschema<4.24,>=4.18 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (4.23.0)
    Requirement already satisfied: seqeval<1.3.0,>=1.2.2 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.2.2)
    Requirement already satisfied: evaluate<0.5.0,>=0.4.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.4.1)
    Requirement already satisfied: timm<1.0.7,>=0.9.5 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.0.3)
    Requirement already satisfied: torchvision<0.22.0,>=0.16.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.21.0)
    Requirement already satisfied: scikit-image<0.26.0,>=0.19.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.25.2)
    Requirement already satisfied: text-unidecode<1.4,>=1.3 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.3)
    Requirement already satisfied: torchmetrics<1.8,>=1.2.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.7.1)
    Requirement already satisfied: omegaconf<2.4.0,>=2.1.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (2.3.0)
    Requirement already satisfied: pytorch-metric-learning<2.9,>=1.3.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (2.8.1)
    Requirement already satisfied: nlpaug<1.2.0,>=1.1.10 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.1.11)
    Requirement already satisfied: nltk<4.0,>=3.4.5 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (3.9.1)
    Requirement already satisfied: openmim<0.4.0,>=0.3.7 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.3.7)
    Requirement already satisfied: defusedxml<0.7.2,>=0.7.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.7.1)
    Requirement already satisfied: jinja2<3.2,>=3.0.3 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (3.1.6)
    Requirement already satisfied: tensorboard<3,>=2.9 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (2.18.0)
    Requirement already satisfied: pytesseract<0.4,>=0.3.9 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (0.3.13)
    Requirement already satisfied: nvidia-ml-py3<8.0,>=7.352.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (7.352.0)
    Requirement already satisfied: pdf2image<1.19,>=1.17.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.multimodal==1.3.0->autogluon) (1.17.0)
    Requirement already satisfied: catboost<1.3,>=1.2 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (1.2.7)
    Requirement already satisfied: einops<0.9,>=0.7 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (0.8.1)
    Requirement already satisfied: spacy<3.9 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (3.8.5)
    Requirement already satisfied: xgboost<3.1,>=2.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (2.1.4)
    Requirement already satisfied: huggingface_hub[torch] in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (0.30.2)
    Requirement already satisfied: lightgbm<4.7,>=4.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (4.6.0)
    Requirement already satisfied: fastai<2.9,>=2.3.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.tabular[all]==1.3.0->autogluon) (2.7.19)
    Requirement already satisfied: joblib<2,>=1.1 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (1.5.0)
    Requirement already satisfied: pytorch_lightning in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.5.1.post0)
    Requirement already satisfied: gluonts<0.17,>=0.15.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.16.1)
    Requirement already satisfied: statsforecast<2.0.2,>=1.7.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.0.1)
    Requirement already satisfied: mlforecast<0.14,>0.13 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.13.6)
    Requirement already satisfied: utilsforecast<0.2.11,>=0.2.3 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.2.10)
    Requirement already satisfied: coreforecast<0.0.16,>=0.0.12 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.0.15)
    Requirement already satisfied: fugue>=0.9.0 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.9.1)
    Requirement already satisfied: orjson~=3.9 in /opt/conda/lib/python3.12/site-packages (from autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (3.10.18)
    Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.12/site-packages (from accelerate<2.0,>=0.34.0->autogluon.multimodal==1.3.0->autogluon) (24.2)
    Requirement already satisfied: pyyaml in /opt/conda/lib/python3.12/site-packages (from accelerate<2.0,>=0.34.0->autogluon.multimodal==1.3.0->autogluon) (6.0.2)
    Requirement already satisfied: safetensors>=0.4.3 in /opt/conda/lib/python3.12/site-packages (from accelerate<2.0,>=0.34.0->autogluon.multimodal==1.3.0->autogluon) (0.5.3)
    Requirement already satisfied: botocore<1.38.0,>=1.37.1 in /opt/conda/lib/python3.12/site-packages (from boto3<2,>=1.10->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.37.1)
    Requirement already satisfied: jmespath<2.0.0,>=0.7.1 in /opt/conda/lib/python3.12/site-packages (from boto3<2,>=1.10->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.0.1)
    Requirement already satisfied: s3transfer<0.12.0,>=0.11.0 in /opt/conda/lib/python3.12/site-packages (from boto3<2,>=1.10->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (0.11.3)
    Requirement already satisfied: python-dateutil<3.0.0,>=2.1 in /opt/conda/lib/python3.12/site-packages (from botocore<1.38.0,>=1.37.1->boto3<2,>=1.10->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2.9.0.post0)
    Requirement already satisfied: urllib3!=2.2.0,<3,>=1.25.4 in /opt/conda/lib/python3.12/site-packages (from botocore<1.38.0,>=1.37.1->boto3<2,>=1.10->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.26.19)
    Requirement already satisfied: graphviz in /opt/conda/lib/python3.12/site-packages (from catboost<1.3,>=1.2->autogluon.tabular[all]==1.3.0->autogluon) (0.20.3)
    Requirement already satisfied: plotly in /opt/conda/lib/python3.12/site-packages (from catboost<1.3,>=1.2->autogluon.tabular[all]==1.3.0->autogluon) (6.0.1)
    Requirement already satisfied: six in /opt/conda/lib/python3.12/site-packages (from catboost<1.3,>=1.2->autogluon.tabular[all]==1.3.0->autogluon) (1.17.0)
    Requirement already satisfied: datasets>=2.0.0 in /opt/conda/lib/python3.12/site-packages (from evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (2.2.1)
    Requirement already satisfied: dill in /opt/conda/lib/python3.12/site-packages (from evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (0.4.0)
    Requirement already satisfied: xxhash in /opt/conda/lib/python3.12/site-packages (from evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (3.5.0)
    Requirement already satisfied: multiprocess in /opt/conda/lib/python3.12/site-packages (from evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (0.70.18)
    Requirement already satisfied: fsspec>=2021.05.0 in /opt/conda/lib/python3.12/site-packages (from fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (2024.10.0)
    Requirement already satisfied: responses<0.19 in /opt/conda/lib/python3.12/site-packages (from evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (0.18.0)
    Requirement already satisfied: pip in /opt/conda/lib/python3.12/site-packages (from fastai<2.9,>=2.3.1->autogluon.tabular[all]==1.3.0->autogluon) (25.1.1)
    Requirement already satisfied: fastdownload<2,>=0.0.5 in /opt/conda/lib/python3.12/site-packages (from fastai<2.9,>=2.3.1->autogluon.tabular[all]==1.3.0->autogluon) (0.0.7)
    Requirement already satisfied: fastcore<1.8,>=1.5.29 in /opt/conda/lib/python3.12/site-packages (from fastai<2.9,>=2.3.1->autogluon.tabular[all]==1.3.0->autogluon) (1.7.20)
    Requirement already satisfied: fastprogress>=0.2.4 in /opt/conda/lib/python3.12/site-packages (from fastai<2.9,>=2.3.1->autogluon.tabular[all]==1.3.0->autogluon) (1.0.3)
    Requirement already satisfied: pydantic<3,>=1.7 in /opt/conda/lib/python3.12/site-packages (from gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.11.4)
    Requirement already satisfied: toolz~=0.10 in /opt/conda/lib/python3.12/site-packages (from gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.12.1)
    Requirement already satisfied: typing-extensions~=4.0 in /opt/conda/lib/python3.12/site-packages (from gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (4.13.2)
    Requirement already satisfied: future in /opt/conda/lib/python3.12/site-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==1.3.0->autogluon) (1.0.0)
    Requirement already satisfied: cloudpickle in /opt/conda/lib/python3.12/site-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==1.3.0->autogluon) (3.1.1)
    Requirement already satisfied: py4j in /opt/conda/lib/python3.12/site-packages (from hyperopt<0.2.8,>=0.2.7->autogluon.core[all]==1.3.0->autogluon) (0.10.9.9)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.12/site-packages (from jinja2<3.2,>=3.0.3->autogluon.multimodal==1.3.0->autogluon) (3.0.2)
    Requirement already satisfied: attrs>=22.2.0 in /opt/conda/lib/python3.12/site-packages (from jsonschema<4.24,>=4.18->autogluon.multimodal==1.3.0->autogluon) (23.2.0)
    Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /opt/conda/lib/python3.12/site-packages (from jsonschema<4.24,>=4.18->autogluon.multimodal==1.3.0->autogluon) (2025.4.1)
    Requirement already satisfied: referencing>=0.28.4 in /opt/conda/lib/python3.12/site-packages (from jsonschema<4.24,>=4.18->autogluon.multimodal==1.3.0->autogluon) (0.36.2)
    Requirement already satisfied: rpds-py>=0.7.1 in /opt/conda/lib/python3.12/site-packages (from jsonschema<4.24,>=4.18->autogluon.multimodal==1.3.0->autogluon) (0.24.0)
    Requirement already satisfied: lightning-utilities<2.0,>=0.10.0 in /opt/conda/lib/python3.12/site-packages (from lightning<2.7,>=2.2->autogluon.multimodal==1.3.0->autogluon) (0.14.3)
    Requirement already satisfied: aiohttp!=4.0.0a0,!=4.0.0a1 in /opt/conda/lib/python3.12/site-packages (from fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (3.9.5)
    Requirement already satisfied: setuptools in /opt/conda/lib/python3.12/site-packages (from lightning-utilities<2.0,>=0.10.0->lightning<2.7,>=2.2->autogluon.multimodal==1.3.0->autogluon) (80.9.0)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.12/site-packages (from matplotlib<3.11,>=3.7.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.3.2)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.12/site-packages (from matplotlib<3.11,>=3.7.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.12/site-packages (from matplotlib<3.11,>=3.7.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (4.57.0)
    Requirement already satisfied: kiwisolver>=1.3.1 in /opt/conda/lib/python3.12/site-packages (from matplotlib<3.11,>=3.7.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (1.4.8)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/lib/python3.12/site-packages (from matplotlib<3.11,>=3.7.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.2.3)
    Requirement already satisfied: numba in /opt/conda/lib/python3.12/site-packages (from mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.61.2)
    Requirement already satisfied: optuna in /opt/conda/lib/python3.12/site-packages (from mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (4.3.0)
    Requirement already satisfied: window-ops in /opt/conda/lib/python3.12/site-packages (from mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.0.15)
    Requirement already satisfied: gdown>=4.0.0 in /opt/conda/lib/python3.12/site-packages (from nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==1.3.0->autogluon) (5.2.0)
    Requirement already satisfied: click in /opt/conda/lib/python3.12/site-packages (from nltk<4.0,>=3.4.5->autogluon.multimodal==1.3.0->autogluon) (8.1.8)
    Requirement already satisfied: regex>=2021.8.3 in /opt/conda/lib/python3.12/site-packages (from nltk<4.0,>=3.4.5->autogluon.multimodal==1.3.0->autogluon) (2024.11.6)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /opt/conda/lib/python3.12/site-packages (from omegaconf<2.4.0,>=2.1.1->autogluon.multimodal==1.3.0->autogluon) (4.9.3)
    Requirement already satisfied: colorama in /opt/conda/lib/python3.12/site-packages (from openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (0.4.6)
    Requirement already satisfied: model-index in /opt/conda/lib/python3.12/site-packages (from openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (0.1.11)
    Requirement already satisfied: rich in /opt/conda/lib/python3.12/site-packages (from openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (14.0.0)
    Requirement already satisfied: tabulate in /opt/conda/lib/python3.12/site-packages (from openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (0.9.0)
    Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.12/site-packages (from pandas<2.3.0,>=2.0.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2024.2)
    Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.12/site-packages (from pandas<2.3.0,>=2.0.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2025.2)
    Requirement already satisfied: annotated-types>=0.6.0 in /opt/conda/lib/python3.12/site-packages (from pydantic<3,>=1.7->gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.7.0)
    Requirement already satisfied: pydantic-core==2.33.2 in /opt/conda/lib/python3.12/site-packages (from pydantic<3,>=1.7->gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.33.2)
    Requirement already satisfied: typing-inspection>=0.4.0 in /opt/conda/lib/python3.12/site-packages (from pydantic<3,>=1.7->gluonts<0.17,>=0.15.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.4.0)
    Requirement already satisfied: charset_normalizer<4,>=2 in /opt/conda/lib/python3.12/site-packages (from requests->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.4.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.12/site-packages (from requests->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.10)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.12/site-packages (from requests->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (2025.4.26)
    Requirement already satisfied: imageio!=2.35.0,>=2.33 in /opt/conda/lib/python3.12/site-packages (from scikit-image<0.26.0,>=0.19.1->autogluon.multimodal==1.3.0->autogluon) (2.37.0)
    Requirement already satisfied: tifffile>=2022.8.12 in /opt/conda/lib/python3.12/site-packages (from scikit-image<0.26.0,>=0.19.1->autogluon.multimodal==1.3.0->autogluon) (2025.3.30)
    Requirement already satisfied: lazy-loader>=0.4 in /opt/conda/lib/python3.12/site-packages (from scikit-image<0.26.0,>=0.19.1->autogluon.multimodal==1.3.0->autogluon) (0.4)
    Requirement already satisfied: threadpoolctl>=3.1.0 in /opt/conda/lib/python3.12/site-packages (from scikit-learn<1.7.0,>=1.4.0->autogluon.core==1.3.0->autogluon.core[all]==1.3.0->autogluon) (3.6.0)
    Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (3.0.12)
    Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.0.5)
    Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.0.10)
    Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (2.0.11)
    Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (3.0.9)
    Requirement already satisfied: thinc<8.4.0,>=8.3.4 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (8.3.4)
    Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.1.3)
    Requirement already satisfied: srsly<3.0.0,>=2.4.3 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (2.5.1)
    Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (2.0.10)
    Requirement already satisfied: weasel<0.5.0,>=0.1.0 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (0.4.1)
    Requirement already satisfied: typer<1.0.0,>=0.3.0 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (0.15.3)
    Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in /opt/conda/lib/python3.12/site-packages (from spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (3.4.1)
    Requirement already satisfied: language-data>=1.2 in /opt/conda/lib/python3.12/site-packages (from langcodes<4.0.0,>=3.2.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: ujson>=1.35 in /opt/conda/lib/python3.12/site-packages (from srsly<3.0.0,>=2.4.3->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (5.10.0)
    Requirement already satisfied: statsmodels>=0.13.2 in /opt/conda/lib/python3.12/site-packages (from statsforecast<2.0.2,>=1.7.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.14.4)
    Requirement already satisfied: absl-py>=0.4 in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (2.2.0)
    Requirement already satisfied: grpcio>=1.48.2 in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (1.67.1)
    Requirement already satisfied: markdown>=2.6.8 in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (3.8)
    Requirement already satisfied: protobuf in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (5.28.3)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (0.7.0)
    Requirement already satisfied: werkzeug>=1.0.1 in /opt/conda/lib/python3.12/site-packages (from tensorboard<3,>=2.9->autogluon.multimodal==1.3.0->autogluon) (3.1.3)
    Requirement already satisfied: blis<1.3.0,>=1.2.0 in /opt/conda/lib/python3.12/site-packages (from thinc<8.4.0,>=8.3.4->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.2.1)
    Requirement already satisfied: confection<1.0.0,>=0.0.1 in /opt/conda/lib/python3.12/site-packages (from thinc<8.4.0,>=8.3.4->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (0.1.5)
    Requirement already satisfied: filelock in /opt/conda/lib/python3.12/site-packages (from torch<2.7,>=2.2->autogluon.multimodal==1.3.0->autogluon) (3.18.0)
    Requirement already satisfied: sympy!=1.13.2,>=1.13.1 in /opt/conda/lib/python3.12/site-packages (from torch<2.7,>=2.2->autogluon.multimodal==1.3.0->autogluon) (1.14.0)
    Requirement already satisfied: tokenizers<0.22,>=0.21 in /opt/conda/lib/python3.12/site-packages (from transformers<4.50,>=4.38.0->transformers[sentencepiece]<4.50,>=4.38.0->autogluon.multimodal==1.3.0->autogluon) (0.21.1)
    Requirement already satisfied: sentencepiece!=0.1.92,>=0.1.91 in /opt/conda/lib/python3.12/site-packages (from transformers[sentencepiece]<4.50,>=4.38.0->autogluon.multimodal==1.3.0->autogluon) (0.2.0)
    Requirement already satisfied: shellingham>=1.3.0 in /opt/conda/lib/python3.12/site-packages (from typer<1.0.0,>=0.3.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.5.4)
    Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in /opt/conda/lib/python3.12/site-packages (from weasel<0.5.0,>=0.1.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (0.21.0)
    Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in /opt/conda/lib/python3.12/site-packages (from weasel<0.5.0,>=0.1.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (7.1.0)
    Requirement already satisfied: wrapt in /opt/conda/lib/python3.12/site-packages (from smart-open<8.0.0,>=5.2.1->weasel<0.5.0,>=0.1.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.17.2)
    Requirement already satisfied: aiosignal>=1.1.2 in /opt/conda/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (1.3.2)
    Requirement already satisfied: frozenlist>=1.1.1 in /opt/conda/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (1.6.0)
    Requirement already satisfied: multidict<7.0,>=4.5 in /opt/conda/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (6.4.3)
    Requirement already satisfied: yarl<2.0,>=1.0 in /opt/conda/lib/python3.12/site-packages (from aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (1.20.0)
    Requirement already satisfied: propcache>=0.2.1 in /opt/conda/lib/python3.12/site-packages (from yarl<2.0,>=1.0->aiohttp!=4.0.0a0,!=4.0.0a1->fsspec[http]>=2021.05.0->evaluate<0.5.0,>=0.4.0->autogluon.multimodal==1.3.0->autogluon) (0.3.1)
    Requirement already satisfied: triad>=0.9.7 in /opt/conda/lib/python3.12/site-packages (from fugue>=0.9.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.9.8)
    Requirement already satisfied: adagio>=0.2.4 in /opt/conda/lib/python3.12/site-packages (from fugue>=0.9.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.2.6)
    Requirement already satisfied: beautifulsoup4 in /opt/conda/lib/python3.12/site-packages (from gdown>=4.0.0->nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==1.3.0->autogluon) (4.13.4)
    Requirement already satisfied: marisa-trie>=1.1.0 in /opt/conda/lib/python3.12/site-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy<3.9->autogluon.tabular[all]==1.3.0->autogluon) (1.2.1)
    Requirement already satisfied: llvmlite<0.45,>=0.44.0dev0 in /opt/conda/lib/python3.12/site-packages (from numba->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (0.44.0)
    Requirement already satisfied: markdown-it-py>=2.2.0 in /opt/conda/lib/python3.12/site-packages (from rich->openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (3.0.0)
    Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /opt/conda/lib/python3.12/site-packages (from rich->openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (2.19.1)
    Requirement already satisfied: mdurl~=0.1 in /opt/conda/lib/python3.12/site-packages (from markdown-it-py>=2.2.0->rich->openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (0.1.2)
    Requirement already satisfied: patsy>=0.5.6 in /opt/conda/lib/python3.12/site-packages (from statsmodels>=0.13.2->statsforecast<2.0.2,>=1.7.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (1.0.1)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/conda/lib/python3.12/site-packages (from sympy!=1.13.2,>=1.13.1->torch<2.7,>=2.2->autogluon.multimodal==1.3.0->autogluon) (1.3.0)
    Requirement already satisfied: fs in /opt/conda/lib/python3.12/site-packages (from triad>=0.9.7->fugue>=0.9.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.4.16)
    Requirement already satisfied: soupsieve>1.2 in /opt/conda/lib/python3.12/site-packages (from beautifulsoup4->gdown>=4.0.0->nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==1.3.0->autogluon) (2.7)
    Requirement already satisfied: appdirs~=1.4.3 in /opt/conda/lib/python3.12/site-packages (from fs->triad>=0.9.7->fugue>=0.9.0->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (1.4.4)
    Requirement already satisfied: ordered-set in /opt/conda/lib/python3.12/site-packages (from model-index->openmim<0.4.0,>=0.3.7->autogluon.multimodal==1.3.0->autogluon) (4.1.0)
    Requirement already satisfied: alembic>=1.5.0 in /opt/conda/lib/python3.12/site-packages (from optuna->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (1.15.2)
    Requirement already satisfied: colorlog in /opt/conda/lib/python3.12/site-packages (from optuna->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (6.9.0)
    Requirement already satisfied: sqlalchemy>=1.4.2 in /opt/conda/lib/python3.12/site-packages (from optuna->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (2.0.40)
    Requirement already satisfied: Mako in /opt/conda/lib/python3.12/site-packages (from alembic>=1.5.0->optuna->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (1.3.10)
    Requirement already satisfied: greenlet>=1 in /opt/conda/lib/python3.12/site-packages (from sqlalchemy>=1.4.2->optuna->mlforecast<0.14,>0.13->autogluon.timeseries==1.3.0->autogluon.timeseries[all]==1.3.0->autogluon) (3.2.2)
    Requirement already satisfied: narwhals>=1.15.1 in /opt/conda/lib/python3.12/site-packages (from plotly->catboost<1.3,>=1.2->autogluon.tabular[all]==1.3.0->autogluon) (1.38.2)
    Requirement already satisfied: PySocks!=1.5.7,>=1.5.6 in /opt/conda/lib/python3.12/site-packages (from requests[socks]->gdown>=4.0.0->nlpaug<1.2.0,>=1.1.10->autogluon.multimodal==1.3.0->autogluon) (1.7.1)


### Setup Kaggle API Key


```python
# create the .kaggle directory and an empty kaggle.json file
!mkdir -p ~/.kaggle
!touch ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
```


```python
# Fill in your user name and key from creating the kaggle account and API token file
import json
import os

kaggle_username = "arbaouimeriem"
kaggle_key = "3bc9d1c137de5725f64c1d29daf3f5a4"

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
# Save API token the kaggle.json file
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```

### Download and explore dataset

### Go to the [bike sharing demand competition](https://www.kaggle.com/c/bike-sharing-demand) and agree to the terms
![kaggle6.png](kaggle6.png)


```python
# Download the dataset, it will be in a .zip file so you'll need to unzip it as well.
#!kaggle competitions download -c bike-sharing-demand
# If you already downloaded it you can use the -o command to overwrite the file
!unzip -o bike-sharing-demand.zip
```

    Archive:  bike-sharing-demand.zip
      inflating: sampleSubmission.csv    
      inflating: test.csv                
      inflating: train.csv               



```python
import pandas as pd
from autogluon.tabular import TabularPredictor
```


```python
# Create the train dataset in pandas by reading the csv
# Set the parsing of the datetime column so you can use some of the `dt` features in pandas later
train = pd.read_csv("train.csv", parse_dates=["datetime"])
# take a look at the first few rows
train.head()
```




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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Simple output of the train dataset to view some of the min/max/varition of the dataset features.
print("********Dataset Info:*********")
train.info()
```

    ********Dataset Info:*********
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     #   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB



```python
# get basic statistical summary
print("********Statistical Summary:********")
print(train.describe())
```

    ********Statistical Summary:********
                                datetime        season       holiday  \
    count                          10886  10886.000000  10886.000000   
    mean   2011-12-27 05:56:22.399411968      2.506614      0.028569   
    min              2011-01-01 00:00:00      1.000000      0.000000   
    25%              2011-07-02 07:15:00      2.000000      0.000000   
    50%              2012-01-01 20:30:00      3.000000      0.000000   
    75%              2012-07-01 12:45:00      4.000000      0.000000   
    max              2012-12-19 23:00:00      4.000000      1.000000   
    std                              NaN      1.116174      0.166599   
    
             workingday       weather         temp         atemp      humidity  \
    count  10886.000000  10886.000000  10886.00000  10886.000000  10886.000000   
    mean       0.680875      1.418427     20.23086     23.655084     61.886460   
    min        0.000000      1.000000      0.82000      0.760000      0.000000   
    25%        0.000000      1.000000     13.94000     16.665000     47.000000   
    50%        1.000000      1.000000     20.50000     24.240000     62.000000   
    75%        1.000000      2.000000     26.24000     31.060000     77.000000   
    max        1.000000      4.000000     41.00000     45.455000    100.000000   
    std        0.466159      0.633839      7.79159      8.474601     19.245033   
    
              windspeed        casual    registered         count  
    count  10886.000000  10886.000000  10886.000000  10886.000000  
    mean      12.799395     36.021955    155.552177    191.574132  
    min        0.000000      0.000000      0.000000      1.000000  
    25%        7.001500      4.000000     36.000000     42.000000  
    50%       12.998000     17.000000    118.000000    145.000000  
    75%       16.997900     49.000000    222.000000    284.000000  
    max       56.996900    367.000000    886.000000    977.000000  
    std        8.164537     49.960477    151.039033    181.144454  



```python
print("********Missing Values:********")
print(train.isnull().sum())
```

    ********Missing Values:********
    datetime      0
    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    casual        0
    registered    0
    count         0
    dtype: int64



```python
# Create the test pandas dataframe in pandas by reading the csv, remember to parse the datetime!
test = pd.read_csv("test.csv", parse_dates=["datetime"])
test.head()
```




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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>13.635</td>
      <td>56</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Same thing as train and test dataset
submission = pd.read_csv("sampleSubmission.csv")
submission.head()
```




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
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Train a model using AutoGluon’s Tabular Prediction

Requirements:
* We are predicting `count`, so it is the label we are setting.
* Ignore `casual` and `registered` columns as they are also not present in the test dataset. 
* Use the `root_mean_squared_error` as the metric to use for evaluation.
* Set a time limit of 10 minutes (600 seconds).
* Use the preset `best_quality` to focus on creating the best model.


```python
# drop casual and registered, as they're not in the test set
train_data = train.drop(columns=["casual", "registered"])

#set up and train the AutoGluon model
predictor = TabularPredictor(label="count", eval_metric="root_mean_squared_error").fit(
    train_data=train_data,
    time_limit=600,
    presets="best_quality"
)

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20250608_201624"
    Verbosity: 2 (Standard Logging)
    =================== System Info ===================
    AutoGluon Version:  1.3.0
    Python Version:     3.12.9
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue May 6 04:10:50 UTC 2025
    CPU Count:          2
    Memory Avail:       1.89 GB / 3.76 GB (50.2%)
    Disk Space Avail:   4.90 GB / 4.99 GB (98.2%)
    	WARNING: Available disk space is low and there is a risk that AutoGluon will run out of disk during fit, causing an exception. 
    	We recommend a minimum available disk space of 10 GB, and large datasets may require more.
    ===================================================
    Presets specified: ['best_quality']
    Setting dynamic_stacking from 'auto' to True. Reason: Enable dynamic_stacking when use_bag_holdout is disabled. (use_bag_holdout=False)
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=1
    DyStack is enabled (dynamic_stacking=True). AutoGluon will try to determine whether the input data is affected by stacked overfitting and enable or disable stacking as a consequence.
    	This is used to identify the optimal `num_stack_levels` value. Copies of AutoGluon will be fit on subsets of the data. Then holdout validation data is used to detect stacked overfitting.
    	Running DyStack for up to 150s of the 600s of remaining time (25%).
    	Running DyStack sub-fit in a ray process to avoid memory leakage. Enabling ray logging (enable_ray_logging=True). Specify `ds_args={'enable_ray_logging': False}` if you experience logging issues.
    2025-06-08 20:16:29,312	WARNING services.py:2070 -- WARNING: The object store is using /tmp instead of /dev/shm because /dev/shm has only 411021312 bytes available. This will harm performance! You may be able to free up space by deleting files in /dev/shm. If you are inside a Docker container, you can increase /dev/shm size by passing '--shm-size=0.91gb' to 'docker run' (or add it to the run_options list in a Ray cluster config). Make sure to set this to more than 30% of available RAM.
    2025-06-08 20:16:29,583	INFO worker.py:1843 -- Started a local Ray instance. View the dashboard at [1m[32mhttp://127.0.0.1:8265 [39m[22m
    		Context path: "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624/ds_sub_fit/sub_fit_ho"
    [36m(_dystack pid=1421)[0m Running DyStack sub-fit ...
    [36m(_dystack pid=1421)[0m /opt/conda/lib/python3.12/site-packages/autogluon/common/utils/utils.py:97: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
    [36m(_dystack pid=1421)[0m   import pkg_resources
    [36m(_dystack pid=1421)[0m Beginning AutoGluon training ... Time limit = 146s
    [36m(_dystack pid=1421)[0m AutoGluon will save models to "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624/ds_sub_fit/sub_fit_ho"
    [36m(_dystack pid=1421)[0m Train Data Rows:    9676
    [36m(_dystack pid=1421)[0m Train Data Columns: 9
    [36m(_dystack pid=1421)[0m Label Column:       count
    [36m(_dystack pid=1421)[0m Problem Type:       regression
    [36m(_dystack pid=1421)[0m Preprocessing data ...
    [36m(_dystack pid=1421)[0m Using Feature Generators to preprocess the data ...
    [36m(_dystack pid=1421)[0m Fitting AutoMLPipelineFeatureGenerator...
    [36m(_dystack pid=1421)[0m 	Available Memory:                    1528.06 MB
    [36m(_dystack pid=1421)[0m 	Train Data (Original)  Memory Usage: 0.66 MB (0.0% of available memory)
    [36m(_dystack pid=1421)[0m 	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    [36m(_dystack pid=1421)[0m 	Stage 1 Generators:
    [36m(_dystack pid=1421)[0m 		Fitting AsTypeFeatureGenerator...
    [36m(_dystack pid=1421)[0m 			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    [36m(_dystack pid=1421)[0m 	Stage 2 Generators:
    [36m(_dystack pid=1421)[0m 		Fitting FillNaFeatureGenerator...
    [36m(_dystack pid=1421)[0m 	Stage 3 Generators:
    [36m(_dystack pid=1421)[0m 		Fitting IdentityFeatureGenerator...
    [36m(_dystack pid=1421)[0m 		Fitting DatetimeFeatureGenerator...
    [36m(_dystack pid=1421)[0m 	Stage 4 Generators:
    [36m(_dystack pid=1421)[0m 		Fitting DropUniqueFeatureGenerator...
    [36m(_dystack pid=1421)[0m 	Stage 5 Generators:
    [36m(_dystack pid=1421)[0m 		Fitting DropDuplicatesFeatureGenerator...
    [36m(_dystack pid=1421)[0m 	Types of features in original data (raw dtype, special dtypes):
    [36m(_dystack pid=1421)[0m 		('datetime', []) : 1 | ['datetime']
    [36m(_dystack pid=1421)[0m 		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    [36m(_dystack pid=1421)[0m 		('int', [])      : 5 | ['season', 'holiday', 'workingday', 'weather', 'humidity']
    [36m(_dystack pid=1421)[0m 	Types of features in processed data (raw dtype, special dtypes):
    [36m(_dystack pid=1421)[0m 		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    [36m(_dystack pid=1421)[0m 		('int', [])                  : 3 | ['season', 'weather', 'humidity']
    [36m(_dystack pid=1421)[0m 		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    [36m(_dystack pid=1421)[0m 		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    [36m(_dystack pid=1421)[0m 	0.1s = Fit runtime
    [36m(_dystack pid=1421)[0m 	9 features in original data used to generate 13 features in processed data.
    [36m(_dystack pid=1421)[0m 	Train Data (Processed) Memory Usage: 0.83 MB (0.1% of available memory)
    [36m(_dystack pid=1421)[0m Data preprocessing and feature engineering runtime = 0.07s ...
    [36m(_dystack pid=1421)[0m AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    [36m(_dystack pid=1421)[0m 	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    [36m(_dystack pid=1421)[0m 	To change this, specify the eval_metric parameter of Predictor()
    [36m(_dystack pid=1421)[0m Large model count detected (112 configs) ... Only displaying the first 3 models of each family. To see all, set `verbosity=3`.
    [36m(_dystack pid=1421)[0m User-specified model hyperparameters to be fit:
    [36m(_dystack pid=1421)[0m {
    [36m(_dystack pid=1421)[0m 	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7}}],
    [36m(_dystack pid=1421)[0m 	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
    [36m(_dystack pid=1421)[0m 	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5}}],
    [36m(_dystack pid=1421)[0m 	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -16}}],
    [36m(_dystack pid=1421)[0m 	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11}}],
    [36m(_dystack pid=1421)[0m 	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    [36m(_dystack pid=1421)[0m 	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    [36m(_dystack pid=1421)[0m 	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    [36m(_dystack pid=1421)[0m }
    [36m(_dystack pid=1421)[0m AutoGluon will fit 2 stack levels (L1 to L2) ...
    [36m(_dystack pid=1421)[0m Fitting 108 L1 models, fit_strategy="sequential" ...
    [36m(_dystack pid=1421)[0m Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 96.93s of the 145.42s of remaining time.
    [36m(_dystack pid=1421)[0m 	-107.445	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	0.02s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.04s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 94.05s of the 142.54s of remaining time.
    [36m(_dystack pid=1421)[0m 	-89.9469	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	0.02s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.04s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 93.98s of the 142.48s of remaining time.
    [36m(_dystack pid=1421)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.74%)


    [36m(_ray_fit pid=1593)[0m [1000]	valid_set's rmse: 129.692
    [36m(_ray_fit pid=1692)[0m [1000]	valid_set's rmse: 132.725[32m [repeated 5x across cluster] (Ray deduplicates logs by default. Set RAY_DEDUP_LOGS=0 to disable log deduplication, or see https://docs.ray.io/en/master/ray-observability/user-guides/configure-logging.html#log-deduplication for more options.)[0m
    [36m(_ray_fit pid=1725)[0m [2000]	valid_set's rmse: 126.702[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=1725)[0m [6000]	valid_set's rmse: 125.468[32m [repeated 6x across cluster][0m
    [36m(_ray_fit pid=1760)[0m [7000]	valid_set's rmse: 132.416[32m [repeated 8x across cluster][0m
    [36m(_ray_fit pid=1801)[0m [1000]	valid_set's rmse: 137.712[32m [repeated 3x across cluster][0m
    [36m(_ray_fit pid=1832)[0m [1000]	valid_set's rmse: 139.958[32m [repeated 4x across cluster][0m
    [36m(_ray_fit pid=1832)[0m [5000]	valid_set's rmse: 137.961[32m [repeated 8x across cluster][0m


    [36m(_dystack pid=1421)[0m 	-131.9758	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	55.76s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	9.49s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: LightGBM_BAG_L1 ... Training model for up to 33.32s of the 81.81s of remaining time.
    [36m(_dystack pid=1421)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.90%)


    [36m(_ray_fit pid=1878)[0m [1000]	valid_set's rmse: 129.274
    [36m(_ray_fit pid=1879)[0m [1000]	valid_set's rmse: 129.285
    [36m(_ray_fit pid=1946)[0m [1000]	valid_set's rmse: 135.098
    [36m(_ray_fit pid=2010)[0m [1000]	valid_set's rmse: 124.896
    [36m(_ray_fit pid=2076)[0m [1000]	valid_set's rmse: 134.479[32m [repeated 2x across cluster][0m


    [36m(_dystack pid=1421)[0m 	-131.8496	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	25.62s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	1.31s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 3.72s of the 52.21s of remaining time.
    [36m(_dystack pid=1421)[0m /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
    [36m(_dystack pid=1421)[0m   warnings.warn(
    [36m(_dystack pid=1421)[0m 	-119.5485	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	13.63s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.62s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: WeightedEnsemble_L2 ... Training model for up to 145.43s of the 37.51s of remaining time.
    [36m(_dystack pid=1421)[0m 	Ensemble Weights: {'KNeighborsDist_BAG_L1': 1.0}
    [36m(_dystack pid=1421)[0m 	-89.9469	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	0.02s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.0s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting 106 L2 models, fit_strategy="sequential" ...
    [36m(_dystack pid=1421)[0m Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 37.49s of the 37.47s of remaining time.
    [36m(_dystack pid=1421)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.86%)


    [36m(_ray_fit pid=2155)[0m [1000]	valid_set's rmse: 69.5159[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2221)[0m [1000]	valid_set's rmse: 73.3059[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2306)[0m [1000]	valid_set's rmse: 75.7703[32m [repeated 3x across cluster][0m
    [36m(_ray_fit pid=2374)[0m [1000]	valid_set's rmse: 71.8442[32m [repeated 2x across cluster][0m


    [36m(_dystack pid=1421)[0m 	-73.7411	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	32.01s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	1.69s	 = Validation runtime
    [36m(_dystack pid=1421)[0m Fitting model: LightGBM_BAG_L2 ... Training model for up to 2.73s of the 2.71s of remaining time.
    [36m(_dystack pid=1421)[0m 	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.97%)
    [36m(_ray_fit pid=2441)[0m 	Ran out of time, early stopping on iteration 141. Best iteration is:
    [36m(_ray_fit pid=2441)[0m 	[133]	valid_set's rmse: 63.6667
    [36m(_ray_fit pid=2504)[0m 	Ran out of time, early stopping on iteration 151. Best iteration is:[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2504)[0m 	[151]	valid_set's rmse: 70.7524[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2570)[0m 	Ran out of time, early stopping on iteration 158. Best iteration is:[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2570)[0m 	[133]	valid_set's rmse: 67.5825[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2641)[0m 	Ran out of time, early stopping on iteration 151. Best iteration is:[32m [repeated 2x across cluster][0m
    [36m(_ray_fit pid=2641)[0m 	[125]	valid_set's rmse: 69.5524[32m [repeated 2x across cluster][0m
    [36m(_dystack pid=1421)[0m 	-68.039	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	19.71s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.11s	 = Validation runtime
    [36m(_dystack pid=1421)[0m WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    [36m(_dystack pid=1421)[0m I0000 00:00:1749413959.252957    1488 chttp2_transport.cc:1182] ipv4:169.255.255.2:32899: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T20:19:19.251960162+00:00"}
    [36m(_dystack pid=1421)[0m Fitting model: WeightedEnsemble_L3 ... Training model for up to 145.43s of the -20.39s of remaining time.
    [36m(_dystack pid=1421)[0m 	Ensemble Weights: {'LightGBM_BAG_L2': 1.0}
    [36m(_dystack pid=1421)[0m 	-68.039	 = Validation score   (-root_mean_squared_error)
    [36m(_dystack pid=1421)[0m 	0.03s	 = Training   runtime
    [36m(_dystack pid=1421)[0m 	0.0s	 = Validation runtime
    [36m(_dystack pid=1421)[0m AutoGluon training complete, total runtime = 165.95s ... Best model: WeightedEnsemble_L3 | Estimated inference throughput: 110.0 rows/s (1210 batch size)
    [36m(_dystack pid=1421)[0m TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624/ds_sub_fit/sub_fit_ho")
    [36m(_dystack pid=1421)[0m Deleting DyStack predictor artifacts (clean_up_fits=True) ...
    [36m(_ray_fit pid=2637)[0m 	Ran out of time, early stopping on iteration 178. Best iteration is:
    [36m(_ray_fit pid=2637)[0m 	[138]	valid_set's rmse: 66.756
    Leaderboard on holdout data (DyStack):
                        model  score_holdout   score_val              eval_metric  pred_time_test  pred_time_val    fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0         LightGBM_BAG_L2     -69.853768  -68.038953  root_mean_squared_error        9.156380      11.607945  114.755275                 0.109275                0.113038          19.707566            2       True          8
    1     WeightedEnsemble_L3     -69.853768  -68.038953  root_mean_squared_error        9.157994      11.609080  114.788626                 0.001613                0.001135           0.033351            3       True          9
    2       LightGBMXT_BAG_L2     -71.976405  -73.741085  root_mean_squared_error       10.591672      13.181058  127.061845                 1.544567                1.686151          32.014136            2       True          7
    3   KNeighborsDist_BAG_L1     -92.031272  -89.946854  root_mean_squared_error        0.126090       0.039161    0.016566                 0.126090                0.039161           0.016566            1       True          2
    4     WeightedEnsemble_L2     -92.031272  -89.946854  root_mean_squared_error        0.127957       0.039696    0.032362                 0.001867                0.000534           0.015797            2       True          6
    5   KNeighborsUnif_BAG_L1    -109.161488 -107.445008  root_mean_squared_error        0.083563       0.038520    0.019477                 0.083563                0.038520           0.019477            1       True          1
    6  RandomForestMSE_BAG_L1    -118.495627 -119.548529  root_mean_squared_error        0.698800       0.618241   13.633145                 0.698800                0.618241          13.633145            1       True          5
    7         LightGBM_BAG_L1    -130.706758 -131.849580  root_mean_squared_error        0.975312       1.310428   25.621646                 0.975312                1.310428          25.621646            1       True          4
    8       LightGBMXT_BAG_L1    -131.068281 -131.975832  root_mean_squared_error        7.163340       9.488557   55.756874                 7.163340                9.488557          55.756874            1       True          3
    	1	 = Optimal   num_stack_levels (Stacked Overfitting Occurred: False)
    	183s	 = DyStack   runtime |	417s	 = Remaining runtime
    Starting main fit with num_stack_levels=1.
    	For future fit calls on this dataset, you can skip DyStack to save time: `predictor.fit(..., dynamic_stacking=False, num_stack_levels=1)`
    /opt/conda/lib/python3.12/site-packages/autogluon/common/utils/utils.py:97: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
      import pkg_resources
    Beginning AutoGluon training ... Time limit = 417s
    AutoGluon will save models to "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624"
    Train Data Rows:    10886
    Train Data Columns: 9
    Label Column:       count
    Problem Type:       regression
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    1518.69 MB
    	Train Data (Original)  Memory Usage: 0.75 MB (0.0% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Stage 5 Generators:
    		Fitting DropDuplicatesFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 5 | ['season', 'holiday', 'workingday', 'weather', 'humidity']
    	Types of features in processed data (raw dtype, special dtypes):
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 3 | ['season', 'weather', 'humidity']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.1s = Fit runtime
    	9 features in original data used to generate 13 features in processed data.
    	Train Data (Processed) Memory Usage: 0.93 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.1s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    Large model count detected (112 configs) ... Only displaying the first 3 models of each family. To see all, set `verbosity=3`.
    User-specified model hyperparameters to be fit:
    {
    	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7}}],
    	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
    	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5}}],
    	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -16}}],
    	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11}}],
    	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    }
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 108 L1 models, fit_strategy="sequential" ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 277.53s of the 416.39s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 274.68s of the 413.54s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 274.58s of the 413.45s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.61%)
    	-131.4609	 = Validation score   (-root_mean_squared_error)
    	55.18s	 = Training   runtime
    	10.45s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 212.87s of the 351.73s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.81%)
    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1749414043.255546    1506 chttp2_transport.cc:1182] ipv4:169.255.255.2:45253: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T20:20:43.255535438+00:00"}
    I0000 00:00:1749414043.983844    1507 chttp2_transport.cc:1182] ipv4:169.255.255.2:34115: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T20:20:43.983839889+00:00"}
    I0000 00:00:1749414052.363366    1514 chttp2_transport.cc:1182] ipv4:169.255.255.2:46529: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {created_time:"2025-06-08T20:20:52.363361248+00:00", http2_error:2, grpc_status:14}
    	-131.0542	 = Validation score   (-root_mean_squared_error)
    	27.93s	 = Training   runtime
    	1.74s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 180.31s of the 319.17s of remaining time.
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-116.5484	 = Validation score   (-root_mean_squared_error)
    	14.55s	 = Training   runtime
    	0.67s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 164.55s of the 303.42s of remaining time.
    	Memory not enough to fit 8 folds in parallel. Will train 2 folds in parallel instead (Estimated 21.66% memory usage per fold, 43.32%/80.00% total).
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=21.66%)
    	-130.6883	 = Validation score   (-root_mean_squared_error)
    	135.33s	 = Training   runtime
    	0.13s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 26.00s of the 164.87s of remaining time.
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-124.6007	 = Validation score   (-root_mean_squared_error)
    	7.38s	 = Training   runtime
    	0.61s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 17.43s of the 156.29s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.58%)
    	-142.506	 = Validation score   (-root_mean_squared_error)
    	40.41s	 = Training   runtime
    	0.32s	 = Validation runtime
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 112.43s of remaining time.
    	Ensemble Weights: {'KNeighborsDist_BAG_L1': 1.0}
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.02s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 106 L2 models, fit_strategy="sequential" ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 112.40s of the 112.37s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=1.21%)
    	-60.4539	 = Validation score   (-root_mean_squared_error)
    	47.77s	 = Training   runtime
    	3.72s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 59.13s of the 59.10s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=0.93%)
    	-55.172	 = Validation score   (-root_mean_squared_error)
    	24.08s	 = Training   runtime
    	0.29s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 31.04s of the 31.01s of remaining time.
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-53.3979	 = Validation score   (-root_mean_squared_error)
    	40.94s	 = Training   runtime
    	0.68s	 = Validation runtime
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.00s of the -11.28s of remaining time.
    	Ensemble Weights: {'RandomForestMSE_BAG_L2': 0.739, 'LightGBM_BAG_L2': 0.217, 'LightGBMXT_BAG_L2': 0.043}
    	-53.1398	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 427.84s ... Best model: WeightedEnsemble_L3 | Estimated inference throughput: 80.5 rows/s (1361 batch size)
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624")


### Review AutoGluon's training run with ranking of models that did the best.



```python
predictor.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                         model   score_val              eval_metric  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -53.139808  root_mean_squared_error      18.684008  393.683526                0.000566           0.029379            3       True         13
    1   RandomForestMSE_BAG_L2  -53.397901  root_mean_squared_error      14.671847  321.797478                0.684902          40.940806            2       True         12
    2          LightGBM_BAG_L2  -55.172044  root_mean_squared_error      14.276984  304.940850                0.290040          24.084178            2       True         11
    3        LightGBMXT_BAG_L2  -60.453943  root_mean_squared_error      17.708499  328.629163                3.721555          47.772491            2       True         10
    4    KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error       0.035517    0.029390                0.035517           0.029390            1       True          2
    5      WeightedEnsemble_L2  -84.125061  root_mean_squared_error       0.036159    0.054193                0.000642           0.024804            2       True          9
    6    KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error       0.035012    0.034572                0.035012           0.034572            1       True          1
    7   RandomForestMSE_BAG_L1 -116.548359  root_mean_squared_error       0.665617   14.554681                0.665617          14.554681            1       True          5
    8     ExtraTreesMSE_BAG_L1 -124.600676  root_mean_squared_error       0.606116    7.380377                0.606116           7.380377            1       True          7
    9          CatBoost_BAG_L1 -130.688296  root_mean_squared_error       0.132524  135.329980                0.132524         135.329980            1       True          6
    10         LightGBM_BAG_L1 -131.054162  root_mean_squared_error       1.743822   27.934454                1.743822          27.934454            1       True          4
    11       LightGBMXT_BAG_L1 -131.460909  root_mean_squared_error      10.451231   55.178684               10.451231          55.178684            1       True          3
    12  NeuralNetFastAI_BAG_L1 -142.505973  root_mean_squared_error       0.317105   40.414535                0.317105          40.414535            1       True          8
    Number of models trained: 13
    Types of models trained:
    {'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_RF', 'StackerEnsembleModel_NNFastAiTabular', 'WeightedEnsembleModel', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_KNN', 'StackerEnsembleModel_LGB'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 3 | ['season', 'weather', 'humidity']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: /home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_201624/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L1': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'LightGBMXT_BAG_L1': -131.46090891834504,
      'LightGBM_BAG_L1': -131.054161598899,
      'RandomForestMSE_BAG_L1': -116.54835939455667,
      'CatBoost_BAG_L1': -130.68829573845144,
      'ExtraTreesMSE_BAG_L1': -124.60067564699747,
      'NeuralNetFastAI_BAG_L1': -142.5059727704572,
      'WeightedEnsemble_L2': -84.12506123181602,
      'LightGBMXT_BAG_L2': -60.45394261613642,
      'LightGBM_BAG_L2': -55.17204444631542,
      'RandomForestMSE_BAG_L2': -53.39790081206772,
      'WeightedEnsemble_L3': -53.13980751521667},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': ['KNeighborsUnif_BAG_L1'],
      'KNeighborsDist_BAG_L1': ['KNeighborsDist_BAG_L1'],
      'LightGBMXT_BAG_L1': ['LightGBMXT_BAG_L1'],
      'LightGBM_BAG_L1': ['LightGBM_BAG_L1'],
      'RandomForestMSE_BAG_L1': ['RandomForestMSE_BAG_L1'],
      'CatBoost_BAG_L1': ['CatBoost_BAG_L1'],
      'ExtraTreesMSE_BAG_L1': ['ExtraTreesMSE_BAG_L1'],
      'NeuralNetFastAI_BAG_L1': ['NeuralNetFastAI_BAG_L1'],
      'WeightedEnsemble_L2': ['WeightedEnsemble_L2'],
      'LightGBMXT_BAG_L2': ['LightGBMXT_BAG_L2'],
      'LightGBM_BAG_L2': ['LightGBM_BAG_L2'],
      'RandomForestMSE_BAG_L2': ['RandomForestMSE_BAG_L2'],
      'WeightedEnsemble_L3': ['WeightedEnsemble_L3']},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.03457188606262207,
      'KNeighborsDist_BAG_L1': 0.029389619827270508,
      'LightGBMXT_BAG_L1': 55.17868375778198,
      'LightGBM_BAG_L1': 27.934454202651978,
      'RandomForestMSE_BAG_L1': 14.554680824279785,
      'CatBoost_BAG_L1': 135.3299798965454,
      'ExtraTreesMSE_BAG_L1': 7.380377292633057,
      'NeuralNetFastAI_BAG_L1': 40.41453504562378,
      'WeightedEnsemble_L2': 0.024803638458251953,
      'LightGBMXT_BAG_L2': 47.77249050140381,
      'LightGBM_BAG_L2': 24.08417797088623,
      'RandomForestMSE_BAG_L2': 40.94080567359924,
      'WeightedEnsemble_L3': 0.02937912940979004},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.035012006759643555,
      'KNeighborsDist_BAG_L1': 0.03551745414733887,
      'LightGBMXT_BAG_L1': 10.451231479644775,
      'LightGBM_BAG_L1': 1.743821620941162,
      'RandomForestMSE_BAG_L1': 0.6656174659729004,
      'CatBoost_BAG_L1': 0.1325240135192871,
      'ExtraTreesMSE_BAG_L1': 0.6061158180236816,
      'NeuralNetFastAI_BAG_L1': 0.3171045780181885,
      'WeightedEnsemble_L2': 0.0006418228149414062,
      'LightGBMXT_BAG_L2': 3.721554756164551,
      'LightGBM_BAG_L2': 0.2900397777557373,
      'RandomForestMSE_BAG_L2': 0.6849024295806885,
      'WeightedEnsemble_L3': 0.0005664825439453125},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None}},
     'leaderboard':                      model   score_val              eval_metric  \
     0      WeightedEnsemble_L3  -53.139808  root_mean_squared_error   
     1   RandomForestMSE_BAG_L2  -53.397901  root_mean_squared_error   
     2          LightGBM_BAG_L2  -55.172044  root_mean_squared_error   
     3        LightGBMXT_BAG_L2  -60.453943  root_mean_squared_error   
     4    KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error   
     5      WeightedEnsemble_L2  -84.125061  root_mean_squared_error   
     6    KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error   
     7   RandomForestMSE_BAG_L1 -116.548359  root_mean_squared_error   
     8     ExtraTreesMSE_BAG_L1 -124.600676  root_mean_squared_error   
     9          CatBoost_BAG_L1 -130.688296  root_mean_squared_error   
     10         LightGBM_BAG_L1 -131.054162  root_mean_squared_error   
     11       LightGBMXT_BAG_L1 -131.460909  root_mean_squared_error   
     12  NeuralNetFastAI_BAG_L1 -142.505973  root_mean_squared_error   
     
         pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  \
     0       18.684008  393.683526                0.000566           0.029379   
     1       14.671847  321.797478                0.684902          40.940806   
     2       14.276984  304.940850                0.290040          24.084178   
     3       17.708499  328.629163                3.721555          47.772491   
     4        0.035517    0.029390                0.035517           0.029390   
     5        0.036159    0.054193                0.000642           0.024804   
     6        0.035012    0.034572                0.035012           0.034572   
     7        0.665617   14.554681                0.665617          14.554681   
     8        0.606116    7.380377                0.606116           7.380377   
     9        0.132524  135.329980                0.132524         135.329980   
     10       1.743822   27.934454                1.743822          27.934454   
     11      10.451231   55.178684               10.451231          55.178684   
     12       0.317105   40.414535                0.317105          40.414535   
     
         stack_level  can_infer  fit_order  
     0             3       True         13  
     1             2       True         12  
     2             2       True         11  
     3             2       True         10  
     4             1       True          2  
     5             2       True          9  
     6             1       True          1  
     7             1       True          5  
     8             1       True          7  
     9             1       True          6  
     10            1       True          4  
     11            1       True          3  
     12            1       True          8  }




```python
# show leaderboard with performance of all models
predictor.leaderboard(silent=True)
```




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
      <th>model</th>
      <th>score_val</th>
      <th>eval_metric</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L3</td>
      <td>-53.139808</td>
      <td>root_mean_squared_error</td>
      <td>18.684008</td>
      <td>393.683526</td>
      <td>0.000566</td>
      <td>0.029379</td>
      <td>3</td>
      <td>True</td>
      <td>13</td>
    </tr>
    <tr>
      <th>1</th>
      <td>RandomForestMSE_BAG_L2</td>
      <td>-53.397901</td>
      <td>root_mean_squared_error</td>
      <td>14.671847</td>
      <td>321.797478</td>
      <td>0.684902</td>
      <td>40.940806</td>
      <td>2</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBM_BAG_L2</td>
      <td>-55.172044</td>
      <td>root_mean_squared_error</td>
      <td>14.276984</td>
      <td>304.940850</td>
      <td>0.290040</td>
      <td>24.084178</td>
      <td>2</td>
      <td>True</td>
      <td>11</td>
    </tr>
    <tr>
      <th>3</th>
      <td>LightGBMXT_BAG_L2</td>
      <td>-60.453943</td>
      <td>root_mean_squared_error</td>
      <td>17.708499</td>
      <td>328.629163</td>
      <td>3.721555</td>
      <td>47.772491</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KNeighborsDist_BAG_L1</td>
      <td>-84.125061</td>
      <td>root_mean_squared_error</td>
      <td>0.035517</td>
      <td>0.029390</td>
      <td>0.035517</td>
      <td>0.029390</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WeightedEnsemble_L2</td>
      <td>-84.125061</td>
      <td>root_mean_squared_error</td>
      <td>0.036159</td>
      <td>0.054193</td>
      <td>0.000642</td>
      <td>0.024804</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>KNeighborsUnif_BAG_L1</td>
      <td>-101.546199</td>
      <td>root_mean_squared_error</td>
      <td>0.035012</td>
      <td>0.034572</td>
      <td>0.035012</td>
      <td>0.034572</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>RandomForestMSE_BAG_L1</td>
      <td>-116.548359</td>
      <td>root_mean_squared_error</td>
      <td>0.665617</td>
      <td>14.554681</td>
      <td>0.665617</td>
      <td>14.554681</td>
      <td>1</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ExtraTreesMSE_BAG_L1</td>
      <td>-124.600676</td>
      <td>root_mean_squared_error</td>
      <td>0.606116</td>
      <td>7.380377</td>
      <td>0.606116</td>
      <td>7.380377</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>9</th>
      <td>CatBoost_BAG_L1</td>
      <td>-130.688296</td>
      <td>root_mean_squared_error</td>
      <td>0.132524</td>
      <td>135.329980</td>
      <td>0.132524</td>
      <td>135.329980</td>
      <td>1</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>10</th>
      <td>LightGBM_BAG_L1</td>
      <td>-131.054162</td>
      <td>root_mean_squared_error</td>
      <td>1.743822</td>
      <td>27.934454</td>
      <td>1.743822</td>
      <td>27.934454</td>
      <td>1</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>11</th>
      <td>LightGBMXT_BAG_L1</td>
      <td>-131.460909</td>
      <td>root_mean_squared_error</td>
      <td>10.451231</td>
      <td>55.178684</td>
      <td>10.451231</td>
      <td>55.178684</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NeuralNetFastAI_BAG_L1</td>
      <td>-142.505973</td>
      <td>root_mean_squared_error</td>
      <td>0.317105</td>
      <td>40.414535</td>
      <td>0.317105</td>
      <td>40.414535</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



### Create predictions from test dataset


```python
# generate predictions from the trained model on the test set
predictions = predictor.predict(test)
predictions.head()
```




    0    22.471901
    1    41.281677
    2    44.445595
    3    48.119881
    4    50.507618
    Name: count, dtype: float32



#### NOTE: Kaggle will reject the submission if we don't set everything to be > 0.


```python
# Describe the `predictions` series to see if there are any negative values
predictions.describe()
```




    count    6493.000000
    mean      100.816277
    std        89.850296
    min         3.041237
    25%        20.977753
    50%        63.550697
    75%       166.719589
    max       365.679016
    Name: count, dtype: float64




```python
# How many negative values do we have?
(predictions < 0).sum()
```




    0




```python
# Set them to zero
# we can clip negative values to 0 (if any) even though we can see that the min is 3 (which is > 0) => we have none
predictions = predictions.clip(lower=0)
```


```python
predictions.describe()
```




    count    6493.000000
    mean      100.816277
    std        89.850296
    min         3.041237
    25%        20.977753
    50%        63.550697
    75%       166.719589
    max       365.679016
    Name: count, dtype: float64



### Set predictions to submission dataframe, save, and submit


```python
# assign predictions to the 'count' column of submission
submission["count"] = predictions

# save the dataFrame as a CSV file (no index!)
submission.to_csv("submission.csv", index=False)
```


```python
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission.csv -m "first raw submission"
```

    /bin/bash: line 1: kaggle: command not found


#### View submission via the command line or in the web browser under the competition's page - `My Submissions`


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

    /bin/bash: line 1: kaggle: command not found


#### Initial score of `: 1.79478`

## Step 4: Exploratory Data Analysis and Creating an additional feature
* Any additional feature will do, but a great suggestion would be to separate out the datetime into hour, day, or month parts.


```python
import matplotlib.pyplot as plt

# Plot histograms for all numeric columns in the training set
train.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()
```


    
![png](output_44_0.png)
    



```python

```


```python
# Extract hour from datetime for both train and test datasets
train['hour'] = train['datetime'].dt.hour
test['hour'] = test['datetime'].dt.hour
```

## Make category types for these so models know they are not just numbers
* AutoGluon originally sees these as ints, but in reality they are int representations of a category.
* Setting the dtype to category will classify these as categories in AutoGluon.


```python
train["season"] = train["season"].astype("category")
train["weather"] = train["weather"].astype("category")

test["season"] = test["season"].astype("category")
test["weather"] = test["weather"].astype("category")
```


```python
# View are new feature
train.head()
```




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
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
# View histogram of all features again now with the hour feature
train.hist(figsize=(15, 10), bins=30)
plt.tight_layout()
plt.show()
```


    
![png](output_50_0.png)
    


## Step 5: Rerun the model with the same settings as before, just with more features


```python
train_data = train.drop(columns=["casual", "registered"]) # as ignored before

#set up and train the AutoGluon model
predictor_new_features = TabularPredictor(label='count', eval_metric='root_mean_squared_error').fit(
    train_data=train_data,
    time_limit=600,
    presets="best_quality"
)

```

    No path specified. Models will be saved in: "AutogluonModels/ag-20250608_210859"
    Verbosity: 2 (Standard Logging)
    =================== System Info ===================
    AutoGluon Version:  1.3.0
    Python Version:     3.12.9
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue May 6 04:10:50 UTC 2025
    CPU Count:          2
    Memory Avail:       1.04 GB / 3.76 GB (27.6%)
    Disk Space Avail:   3.90 GB / 4.99 GB (78.1%)
    	WARNING: Available disk space is low and there is a risk that AutoGluon will run out of disk during fit, causing an exception. 
    	We recommend a minimum available disk space of 10 GB, and large datasets may require more.
    ===================================================
    Presets specified: ['best_quality']
    Setting dynamic_stacking from 'auto' to True. Reason: Enable dynamic_stacking when use_bag_holdout is disabled. (use_bag_holdout=False)
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=1
    DyStack is enabled (dynamic_stacking=True). AutoGluon will try to determine whether the input data is affected by stacked overfitting and enable or disable stacking as a consequence.
    	This is used to identify the optimal `num_stack_levels` value. Copies of AutoGluon will be fit on subsets of the data. Then holdout validation data is used to detect stacked overfitting.
    	Running DyStack for up to 150s of the 600s of remaining time (25%).
    		Context path: "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_210859/ds_sub_fit/sub_fit_ho"
    Leaderboard on holdout data (DyStack):
                        model  score_holdout   score_val              eval_metric  pred_time_test  pred_time_val    fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0     WeightedEnsemble_L3     -36.686519  -36.652669  root_mean_squared_error        1.451723       0.255373   58.557416                 0.002360                0.000940           0.040120            3       True         10
    1     WeightedEnsemble_L2     -36.686519  -36.652669  root_mean_squared_error        1.452519       0.254967   58.539987                 0.003156                0.000534           0.022691            2       True          7
    2         CatBoost_BAG_L1     -37.665521  -39.020062  root_mean_squared_error        1.036508       0.035193   53.933459                 1.036508                0.035193          53.933459            1       True          6
    3  RandomForestMSE_BAG_L1     -40.171064  -39.518605  root_mean_squared_error        0.337169       0.173019    4.568396                 0.337169                0.173019           4.568396            1       True          5
    4   KNeighborsDist_BAG_L1     -92.031272  -89.946854  root_mean_squared_error        0.075687       0.046221    0.015442                 0.075687                0.046221           0.015442            1       True          2
    5   KNeighborsUnif_BAG_L1    -109.161488 -107.445008  root_mean_squared_error        0.085802       0.042781    0.017254                 0.085802                0.042781           0.017254            1       True          1
    6         LightGBM_BAG_L2    -173.012044 -172.426320  root_mean_squared_error        3.081555       0.402454  113.841576                 0.025083                0.038274          18.085189            2       True          9
    7       LightGBMXT_BAG_L2    -173.889890 -173.335040  root_mean_squared_error        3.086249       0.391422  113.947895                 0.029777                0.027242          18.191508            2       True          8
    8         LightGBM_BAG_L1    -174.224120 -173.675134  root_mean_squared_error        0.025615       0.041365   18.190923                 0.025615                0.041365          18.190923            1       True          4
    9       LightGBMXT_BAG_L1    -177.885111 -177.352728  root_mean_squared_error        1.495692       0.025601   19.030912                 1.495692                0.025601          19.030912            1       True          3
    	1	 = Optimal   num_stack_levels (Stacked Overfitting Occurred: False)
    	162s	 = DyStack   runtime |	438s	 = Remaining runtime
    Starting main fit with num_stack_levels=1.
    	For future fit calls on this dataset, you can skip DyStack to save time: `predictor.fit(..., dynamic_stacking=False, num_stack_levels=1)`
    Beginning AutoGluon training ... Time limit = 438s
    AutoGluon will save models to "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_210859"
    Train Data Rows:    10886
    Train Data Columns: 10
    Label Column:       count
    Problem Type:       regression
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    724.63 MB
    	Train Data (Original)  Memory Usage: 0.64 MB (0.1% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Stage 5 Generators:
    		Fitting DropDuplicatesFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 4 | ['holiday', 'workingday', 'humidity', 'hour']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 2 | ['humidity', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.1s = Fit runtime
    	10 features in original data used to generate 14 features in processed data.
    	Train Data (Processed) Memory Usage: 0.83 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.12s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    Large model count detected (112 configs) ... Only displaying the first 3 models of each family. To see all, set `verbosity=3`.
    User-specified model hyperparameters to be fit:
    {
    	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -7}}],
    	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
    	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -5}}],
    	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -16}}],
    	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -11}}],
    	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
    	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
    }
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 108 L1 models, fit_strategy="sequential" ...
    Fitting model: KNeighborsUnif_BAG_L1 ... Training model for up to 291.90s of the 437.95s of remaining time.
    	-101.5462	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.05s	 = Validation runtime
    Fitting model: KNeighborsDist_BAG_L1 ... Training model for up to 291.79s of the 437.84s of remaining time.
    	-84.1251	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.05s	 = Validation runtime
    Fitting model: LightGBMXT_BAG_L1 ... Training model for up to 291.69s of the 437.74s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=1.50%)
    I0000 00:00:1749417116.228484    1507 chttp2_transport.cc:1182] ipv4:169.255.255.2:34253: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T21:11:56.228471669+00:00"}
    I0000 00:00:1749417153.775823    1506 chttp2_transport.cc:1182] ipv4:169.255.255.2:38201: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T21:12:33.775819036+00:00"}
    	-34.471	 = Validation score   (-root_mean_squared_error)
    	74.98s	 = Training   runtime
    	15.5s	 = Validation runtime
    Fitting model: LightGBM_BAG_L1 ... Training model for up to 211.65s of the 357.70s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=1.34%)
    	-33.9196	 = Validation score   (-root_mean_squared_error)
    	37.11s	 = Training   runtime
    	3.63s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L1 ... Training model for up to 169.70s of the 315.75s of remaining time.
    	Warning: Reducing model 'n_estimators' from 300 -> 139 due to low memory. Expected memory usage reduced from 32.17% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-38.6678	 = Validation score   (-root_mean_squared_error)
    	7.81s	 = Training   runtime
    	0.44s	 = Validation runtime
    Fitting model: CatBoost_BAG_L1 ... Training model for up to 161.22s of the 307.28s of remaining time.
    	Memory not enough to fit 8 folds in parallel. Will train 2 folds in parallel instead (Estimated 39.16% memory usage per fold, 78.32%/80.00% total).
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=39.16%)
    	-34.5427	 = Validation score   (-root_mean_squared_error)
    	140.68s	 = Training   runtime
    	0.12s	 = Validation runtime
    Fitting model: ExtraTreesMSE_BAG_L1 ... Training model for up to 17.47s of the 163.52s of remaining time.
    	Warning: Reducing model 'n_estimators' from 300 -> 103 due to low memory. Expected memory usage reduced from 43.67% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-39.03	 = Validation score   (-root_mean_squared_error)
    	2.95s	 = Training   runtime
    	0.3s	 = Validation runtime
    Fitting model: NeuralNetFastAI_BAG_L1 ... Training model for up to 14.04s of the 160.09s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=1.48%)
    	-103.6512	 = Validation score   (-root_mean_squared_error)
    	39.63s	 = Training   runtime
    	0.36s	 = Validation runtime
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 116.59s of remaining time.
    	Ensemble Weights: {'LightGBMXT_BAG_L1': 0.32, 'LightGBM_BAG_L1': 0.32, 'CatBoost_BAG_L1': 0.2, 'RandomForestMSE_BAG_L1': 0.12, 'KNeighborsDist_BAG_L1': 0.04}
    	-32.276	 = Validation score   (-root_mean_squared_error)
    	0.03s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 106 L2 models, fit_strategy="sequential" ...
    Fitting model: LightGBMXT_BAG_L2 ... Training model for up to 116.52s of the 116.49s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=3.22%)
    I0000 00:00:1749417433.410542    1514 chttp2_transport.cc:1182] ipv4:169.255.255.2:36255: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T21:17:13.408658847+00:00"}
    I0000 00:00:1749417443.193952    1514 chttp2_transport.cc:1182] ipv4:169.255.255.2:41773: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {created_time:"2025-06-08T21:17:23.193947408+00:00", http2_error:2, grpc_status:14}
    	-31.2686	 = Validation score   (-root_mean_squared_error)
    	30.33s	 = Training   runtime
    	1.37s	 = Validation runtime
    Fitting model: LightGBM_BAG_L2 ... Training model for up to 79.72s of the 79.69s of remaining time.
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (2 workers, per: cpus=1, gpus=0, memory=2.27%)
    I0000 00:00:1749417481.455507    1510 chttp2_transport.cc:1182] ipv4:169.255.255.2:37585: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {created_time:"2025-06-08T21:18:01.455503382+00:00", http2_error:2, grpc_status:14}
    	-30.5997	 = Validation score   (-root_mean_squared_error)
    	25.79s	 = Training   runtime
    	0.38s	 = Validation runtime
    Fitting model: RandomForestMSE_BAG_L2 ... Training model for up to 49.70s of the 49.67s of remaining time.
    I0000 00:00:1749417490.138994    1513 chttp2_transport.cc:1182] ipv4:169.255.255.2:39305: Got goaway [2] err=UNAVAILABLE:GOAWAY received; Error code: 2; Debug Text: Cancelling all calls {grpc_status:14, http2_error:2, created_time:"2025-06-08T21:18:10.138989091+00:00"}
    	Warning: Reducing model 'n_estimators' from 300 -> 147 due to low memory. Expected memory usage reduced from 30.57% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    	-31.8921	 = Validation score   (-root_mean_squared_error)
    	22.22s	 = Training   runtime
    	0.35s	 = Validation runtime
    Fitting model: CatBoost_BAG_L2 ... Training model for up to 26.94s of the 26.91s of remaining time.
    	Memory not enough to fit 8 folds in parallel. Will train 1 folds in parallel instead (Estimated 40.86% memory usage per fold, 40.86%/80.00% total).
    	Fitting 8 child models (S1F1 - S1F8) | Fitting with ParallelLocalFoldFittingStrategy (1 workers, per: cpus=1, gpus=0, memory=40.86%)
    		Switching to pseudo sequential ParallelFoldFittingStrategy to avoid Python memory leakage.
    		Overrule this behavior by setting fold_fitting_strategy to 'sequential_local' in ag_args_ensemble when when calling `predictor.fit`
    	-31.5117	 = Validation score   (-root_mean_squared_error)
    	38.56s	 = Training   runtime
    	0.04s	 = Validation runtime
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.00s of the -14.01s of remaining time.
    	Ensemble Weights: {'LightGBM_BAG_L2': 0.583, 'LightGBMXT_BAG_L2': 0.25, 'CatBoost_BAG_L2': 0.083, 'CatBoost_BAG_L1': 0.042, 'RandomForestMSE_BAG_L2': 0.042}
    	-30.3792	 = Validation score   (-root_mean_squared_error)
    	0.04s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 452.16s ... Best model: WeightedEnsemble_L3 | Estimated inference throughput: 63.2 rows/s (1361 batch size)
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_210859")



```python
predictor_new_features.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                         model   score_val              eval_metric  pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0      WeightedEnsemble_L3  -30.379192  root_mean_squared_error      22.589396  420.163936                0.000597           0.035202            3       True         14
    1          LightGBM_BAG_L2  -30.599703  root_mean_squared_error      20.830935  329.012496                0.380237          25.789837            2       True         11
    2        LightGBMXT_BAG_L2  -31.268551  root_mean_squared_error      21.818809  333.556079                1.368110          30.333420            2       True         10
    3          CatBoost_BAG_L2  -31.511693  root_mean_squared_error      20.486796  341.785052                0.036098          38.562393            2       True         13
    4   RandomForestMSE_BAG_L2  -31.892123  root_mean_squared_error      20.804354  325.443084                0.353656          22.220425            2       True         12
    5      WeightedEnsemble_L2  -32.275971  root_mean_squared_error      19.735300  260.638291                0.000668           0.029859            2       True          9
    6          LightGBM_BAG_L1  -33.919639  root_mean_squared_error       3.628603   37.107718                3.628603          37.107718            1       True          4
    7        LightGBMXT_BAG_L1  -34.470975  root_mean_squared_error      15.498230   74.984527               15.498230          74.984527            1       True          3
    8          CatBoost_BAG_L1  -34.542681  root_mean_squared_error       0.122128  140.677964                0.122128         140.677964            1       True          6
    9   RandomForestMSE_BAG_L1  -38.667821  root_mean_squared_error       0.440506    7.811553                0.440506           7.811553            1       True          5
    10    ExtraTreesMSE_BAG_L1  -39.029968  root_mean_squared_error       0.303425    2.954540                0.303425           2.954540            1       True          7
    11   KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error       0.045164    0.026671                0.045164           0.026671            1       True          2
    12   KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error       0.049525    0.026600                0.049525           0.026600            1       True          1
    13  NeuralNetFastAI_BAG_L1 -103.651152  root_mean_squared_error       0.363117   39.633087                0.363117          39.633087            1       True          8
    Number of models trained: 14
    Types of models trained:
    {'StackerEnsembleModel_CatBoost', 'StackerEnsembleModel_RF', 'StackerEnsembleModel_NNFastAiTabular', 'WeightedEnsembleModel', 'StackerEnsembleModel_XT', 'StackerEnsembleModel_KNN', 'StackerEnsembleModel_LGB'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])             : 2 | ['season', 'weather']
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 2 | ['humidity', 'hour']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: /home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_210859/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'KNeighborsUnif_BAG_L1': 'StackerEnsembleModel_KNN',
      'KNeighborsDist_BAG_L1': 'StackerEnsembleModel_KNN',
      'LightGBMXT_BAG_L1': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L1': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L1': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L1': 'StackerEnsembleModel_CatBoost',
      'ExtraTreesMSE_BAG_L1': 'StackerEnsembleModel_XT',
      'NeuralNetFastAI_BAG_L1': 'StackerEnsembleModel_NNFastAiTabular',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'LightGBMXT_BAG_L2': 'StackerEnsembleModel_LGB',
      'LightGBM_BAG_L2': 'StackerEnsembleModel_LGB',
      'RandomForestMSE_BAG_L2': 'StackerEnsembleModel_RF',
      'CatBoost_BAG_L2': 'StackerEnsembleModel_CatBoost',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'KNeighborsUnif_BAG_L1': -101.54619908446061,
      'KNeighborsDist_BAG_L1': -84.12506123181602,
      'LightGBMXT_BAG_L1': -34.47097500967876,
      'LightGBM_BAG_L1': -33.919639163586254,
      'RandomForestMSE_BAG_L1': -38.667820698814474,
      'CatBoost_BAG_L1': -34.542681129110406,
      'ExtraTreesMSE_BAG_L1': -39.02996846619364,
      'NeuralNetFastAI_BAG_L1': -103.65115155785602,
      'WeightedEnsemble_L2': -32.27597123171691,
      'LightGBMXT_BAG_L2': -31.268550694957717,
      'LightGBM_BAG_L2': -30.59970346775722,
      'RandomForestMSE_BAG_L2': -31.892122506155072,
      'CatBoost_BAG_L2': -31.51169266826183,
      'WeightedEnsemble_L3': -30.3791915482076},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'KNeighborsUnif_BAG_L1': ['KNeighborsUnif_BAG_L1'],
      'KNeighborsDist_BAG_L1': ['KNeighborsDist_BAG_L1'],
      'LightGBMXT_BAG_L1': ['LightGBMXT_BAG_L1'],
      'LightGBM_BAG_L1': ['LightGBM_BAG_L1'],
      'RandomForestMSE_BAG_L1': ['RandomForestMSE_BAG_L1'],
      'CatBoost_BAG_L1': ['CatBoost_BAG_L1'],
      'ExtraTreesMSE_BAG_L1': ['ExtraTreesMSE_BAG_L1'],
      'NeuralNetFastAI_BAG_L1': ['NeuralNetFastAI_BAG_L1'],
      'WeightedEnsemble_L2': ['WeightedEnsemble_L2'],
      'LightGBMXT_BAG_L2': ['LightGBMXT_BAG_L2'],
      'LightGBM_BAG_L2': ['LightGBM_BAG_L2'],
      'RandomForestMSE_BAG_L2': ['RandomForestMSE_BAG_L2'],
      'CatBoost_BAG_L2': ['CatBoost_BAG_L2'],
      'WeightedEnsemble_L3': ['WeightedEnsemble_L3']},
     'model_fit_times': {'KNeighborsUnif_BAG_L1': 0.026600360870361328,
      'KNeighborsDist_BAG_L1': 0.026671171188354492,
      'LightGBMXT_BAG_L1': 74.98452687263489,
      'LightGBM_BAG_L1': 37.107717990875244,
      'RandomForestMSE_BAG_L1': 7.81155252456665,
      'CatBoost_BAG_L1': 140.6779637336731,
      'ExtraTreesMSE_BAG_L1': 2.9545397758483887,
      'NeuralNetFastAI_BAG_L1': 39.633086919784546,
      'WeightedEnsemble_L2': 0.029858827590942383,
      'LightGBMXT_BAG_L2': 30.333420038223267,
      'LightGBM_BAG_L2': 25.7898371219635,
      'RandomForestMSE_BAG_L2': 22.22042489051819,
      'CatBoost_BAG_L2': 38.56239295005798,
      'WeightedEnsemble_L3': 0.0352015495300293},
     'model_pred_times': {'KNeighborsUnif_BAG_L1': 0.049524545669555664,
      'KNeighborsDist_BAG_L1': 0.045163631439208984,
      'LightGBMXT_BAG_L1': 15.49822998046875,
      'LightGBM_BAG_L1': 3.628603458404541,
      'RandomForestMSE_BAG_L1': 0.4405059814453125,
      'CatBoost_BAG_L1': 0.12212848663330078,
      'ExtraTreesMSE_BAG_L1': 0.3034250736236572,
      'NeuralNetFastAI_BAG_L1': 0.3631174564361572,
      'WeightedEnsemble_L2': 0.0006682872772216797,
      'LightGBMXT_BAG_L2': 1.368110179901123,
      'LightGBM_BAG_L2': 0.3802366256713867,
      'RandomForestMSE_BAG_L2': 0.3536558151245117,
      'CatBoost_BAG_L2': 0.03609776496887207,
      'WeightedEnsemble_L3': 0.0005974769592285156},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'KNeighborsUnif_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'KNeighborsDist_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'LightGBMXT_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBM_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'RandomForestMSE_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'CatBoost_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'ExtraTreesMSE_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'NeuralNetFastAI_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBMXT_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'LightGBM_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'RandomForestMSE_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'CatBoost_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None}},
     'leaderboard':                      model   score_val              eval_metric  \
     0      WeightedEnsemble_L3  -30.379192  root_mean_squared_error   
     1          LightGBM_BAG_L2  -30.599703  root_mean_squared_error   
     2        LightGBMXT_BAG_L2  -31.268551  root_mean_squared_error   
     3          CatBoost_BAG_L2  -31.511693  root_mean_squared_error   
     4   RandomForestMSE_BAG_L2  -31.892123  root_mean_squared_error   
     5      WeightedEnsemble_L2  -32.275971  root_mean_squared_error   
     6          LightGBM_BAG_L1  -33.919639  root_mean_squared_error   
     7        LightGBMXT_BAG_L1  -34.470975  root_mean_squared_error   
     8          CatBoost_BAG_L1  -34.542681  root_mean_squared_error   
     9   RandomForestMSE_BAG_L1  -38.667821  root_mean_squared_error   
     10    ExtraTreesMSE_BAG_L1  -39.029968  root_mean_squared_error   
     11   KNeighborsDist_BAG_L1  -84.125061  root_mean_squared_error   
     12   KNeighborsUnif_BAG_L1 -101.546199  root_mean_squared_error   
     13  NeuralNetFastAI_BAG_L1 -103.651152  root_mean_squared_error   
     
         pred_time_val    fit_time  pred_time_val_marginal  fit_time_marginal  \
     0       22.589396  420.163936                0.000597           0.035202   
     1       20.830935  329.012496                0.380237          25.789837   
     2       21.818809  333.556079                1.368110          30.333420   
     3       20.486796  341.785052                0.036098          38.562393   
     4       20.804354  325.443084                0.353656          22.220425   
     5       19.735300  260.638291                0.000668           0.029859   
     6        3.628603   37.107718                3.628603          37.107718   
     7       15.498230   74.984527               15.498230          74.984527   
     8        0.122128  140.677964                0.122128         140.677964   
     9        0.440506    7.811553                0.440506           7.811553   
     10       0.303425    2.954540                0.303425           2.954540   
     11       0.045164    0.026671                0.045164           0.026671   
     12       0.049525    0.026600                0.049525           0.026600   
     13       0.363117   39.633087                0.363117          39.633087   
     
         stack_level  can_infer  fit_order  
     0             3       True         14  
     1             2       True         11  
     2             2       True         10  
     3             2       True         13  
     4             2       True         12  
     5             2       True          9  
     6             1       True          4  
     7             1       True          3  
     8             1       True          6  
     9             1       True          5  
     10            1       True          7  
     11            1       True          2  
     12            1       True          1  
     13            1       True          8  }




```python
predictor_new_features.leaderboard(silent=True)
```




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
      <th>model</th>
      <th>score_val</th>
      <th>eval_metric</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L3</td>
      <td>-30.379192</td>
      <td>root_mean_squared_error</td>
      <td>22.589396</td>
      <td>420.163936</td>
      <td>0.000597</td>
      <td>0.035202</td>
      <td>3</td>
      <td>True</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LightGBM_BAG_L2</td>
      <td>-30.599703</td>
      <td>root_mean_squared_error</td>
      <td>20.830935</td>
      <td>329.012496</td>
      <td>0.380237</td>
      <td>25.789837</td>
      <td>2</td>
      <td>True</td>
      <td>11</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LightGBMXT_BAG_L2</td>
      <td>-31.268551</td>
      <td>root_mean_squared_error</td>
      <td>21.818809</td>
      <td>333.556079</td>
      <td>1.368110</td>
      <td>30.333420</td>
      <td>2</td>
      <td>True</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CatBoost_BAG_L2</td>
      <td>-31.511693</td>
      <td>root_mean_squared_error</td>
      <td>20.486796</td>
      <td>341.785052</td>
      <td>0.036098</td>
      <td>38.562393</td>
      <td>2</td>
      <td>True</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RandomForestMSE_BAG_L2</td>
      <td>-31.892123</td>
      <td>root_mean_squared_error</td>
      <td>20.804354</td>
      <td>325.443084</td>
      <td>0.353656</td>
      <td>22.220425</td>
      <td>2</td>
      <td>True</td>
      <td>12</td>
    </tr>
    <tr>
      <th>5</th>
      <td>WeightedEnsemble_L2</td>
      <td>-32.275971</td>
      <td>root_mean_squared_error</td>
      <td>19.735300</td>
      <td>260.638291</td>
      <td>0.000668</td>
      <td>0.029859</td>
      <td>2</td>
      <td>True</td>
      <td>9</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LightGBM_BAG_L1</td>
      <td>-33.919639</td>
      <td>root_mean_squared_error</td>
      <td>3.628603</td>
      <td>37.107718</td>
      <td>3.628603</td>
      <td>37.107718</td>
      <td>1</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LightGBMXT_BAG_L1</td>
      <td>-34.470975</td>
      <td>root_mean_squared_error</td>
      <td>15.498230</td>
      <td>74.984527</td>
      <td>15.498230</td>
      <td>74.984527</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>8</th>
      <td>CatBoost_BAG_L1</td>
      <td>-34.542681</td>
      <td>root_mean_squared_error</td>
      <td>0.122128</td>
      <td>140.677964</td>
      <td>0.122128</td>
      <td>140.677964</td>
      <td>1</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>9</th>
      <td>RandomForestMSE_BAG_L1</td>
      <td>-38.667821</td>
      <td>root_mean_squared_error</td>
      <td>0.440506</td>
      <td>7.811553</td>
      <td>0.440506</td>
      <td>7.811553</td>
      <td>1</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ExtraTreesMSE_BAG_L1</td>
      <td>-39.029968</td>
      <td>root_mean_squared_error</td>
      <td>0.303425</td>
      <td>2.954540</td>
      <td>0.303425</td>
      <td>2.954540</td>
      <td>1</td>
      <td>True</td>
      <td>7</td>
    </tr>
    <tr>
      <th>11</th>
      <td>KNeighborsDist_BAG_L1</td>
      <td>-84.125061</td>
      <td>root_mean_squared_error</td>
      <td>0.045164</td>
      <td>0.026671</td>
      <td>0.045164</td>
      <td>0.026671</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>KNeighborsUnif_BAG_L1</td>
      <td>-101.546199</td>
      <td>root_mean_squared_error</td>
      <td>0.049525</td>
      <td>0.026600</td>
      <td>0.049525</td>
      <td>0.026600</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NeuralNetFastAI_BAG_L1</td>
      <td>-103.651152</td>
      <td>root_mean_squared_error</td>
      <td>0.363117</td>
      <td>39.633087</td>
      <td>0.363117</td>
      <td>39.633087</td>
      <td>1</td>
      <td>True</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remember to set all negative values to zero
predictions_new_features = predictor_new_features.predict(test)
predictions_new_features = predictions_new_features.clip(lower=0)
```


```python
# Same submitting predictions
submission_new_features = submission.copy()
submission_new_features["count"] = predictions_new_features
submission_new_features.to_csv("submission_new_features.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_features.csv -m "new features"
```


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

#### New Score of `0.62064`

## Step 6: Hyper parameter optimization
* There are many options for hyper parameter optimization.
* Options are to change the AutoGluon higher level parameters or the individual model hyperparameters.
* The hyperparameters of the models themselves that are in AutoGluon. Those need the `hyperparameter` and `hyperparameter_tune_kwargs` arguments.


```python
predictor_new_hpo = TabularPredictor(label='count', eval_metric='root_mean_squared_error').fit(
    train_data=train_data,
    time_limit=600,
    presets='best_quality',
    hyperparameters={
        'GBM': {},  # enable tuning for LightGBM
        'RF': {},   # enable tuning for RandomForest
        'XT': {},   # enable tuning for ExtraTrees
        'NN_TORCH': {},  # neural nets
    },
    hyperparameter_tune_kwargs={
        'num_trials': 10,   # number of HPO trials
        'scheduler': 'local',
        'searcher': 'bayesopt'
    }
)
```

    No path specified. Models will be saved in: "AutogluonModels/ag-20250608_214930"
    Verbosity: 2 (Standard Logging)
    =================== System Info ===================
    AutoGluon Version:  1.3.0
    Python Version:     3.12.9
    Operating System:   Linux
    Platform Machine:   x86_64
    Platform Version:   #1 SMP Tue May 6 04:10:50 UTC 2025
    CPU Count:          2
    Memory Avail:       1.02 GB / 3.76 GB (27.1%)
    Disk Space Avail:   3.36 GB / 4.99 GB (67.3%)
    	WARNING: Available disk space is low and there is a risk that AutoGluon will run out of disk during fit, causing an exception. 
    	We recommend a minimum available disk space of 10 GB, and large datasets may require more.
    ===================================================
    Presets specified: ['best_quality']
    Warning: hyperparameter tuning is currently experimental and may cause the process to hang.
    Setting dynamic_stacking from 'auto' to True. Reason: Enable dynamic_stacking when use_bag_holdout is disabled. (use_bag_holdout=False)
    Stack configuration (auto_stack=True): num_stack_levels=1, num_bag_folds=8, num_bag_sets=1
    DyStack is enabled (dynamic_stacking=True). AutoGluon will try to determine whether the input data is affected by stacked overfitting and enable or disable stacking as a consequence.
    	This is used to identify the optimal `num_stack_levels` value. Copies of AutoGluon will be fit on subsets of the data. Then holdout validation data is used to detect stacked overfitting.
    	Running DyStack for up to 150s of the 600s of remaining time (25%).
    		Context path: "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_214930/ds_sub_fit/sub_fit_ho"
    Leaderboard on holdout data (DyStack):
                     model  score_holdout  score_val              eval_metric  pred_time_test  pred_time_val   fit_time  pred_time_test_marginal  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0    ExtraTrees_BAG_L2     -36.726174 -38.145581  root_mean_squared_error        2.543877       0.897975  16.578194                 0.832046                0.288832           4.567364            2       True          5
    1  WeightedEnsemble_L3     -36.899976 -37.540402  root_mean_squared_error        3.723903       1.207139  27.941968                 0.003114                0.000808           0.020520            3       True          6
    2  RandomForest_BAG_L2     -36.960266 -38.586555  root_mean_squared_error        2.888743       0.917499  23.354084                 1.176912                0.308355          11.343254            2       True          4
    3    ExtraTrees_BAG_L1     -38.063356 -39.719274  root_mean_squared_error        1.022118       0.290696   3.904184                 1.022118                0.290696           3.904184            1       True          2
    4  WeightedEnsemble_L2     -38.095307 -38.024585  root_mean_squared_error        1.722106       0.609705  12.020122                 0.010274                0.000562           0.009291            2       True          3
    5  RandomForest_BAG_L1     -40.139391 -39.117765  root_mean_squared_error        0.689714       0.318447   8.106646                 0.689714                0.318447           8.106646            1       True          1
    	1	 = Optimal   num_stack_levels (Stacked Overfitting Occurred: False)
    	38s	 = DyStack   runtime |	562s	 = Remaining runtime
    Starting main fit with num_stack_levels=1.
    	For future fit calls on this dataset, you can skip DyStack to save time: `predictor.fit(..., dynamic_stacking=False, num_stack_levels=1)`
    Beginning AutoGluon training ... Time limit = 562s
    AutoGluon will save models to "/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_214930"
    Train Data Rows:    10886
    Train Data Columns: 10
    Label Column:       count
    Problem Type:       regression
    Preprocessing data ...
    Using Feature Generators to preprocess the data ...
    Fitting AutoMLPipelineFeatureGenerator...
    	Available Memory:                    728.98 MB
    	Train Data (Original)  Memory Usage: 0.64 MB (0.1% of available memory)
    	Inferring data type of each feature based on column values. Set feature_metadata_in to manually specify special dtypes of the features.
    	Stage 1 Generators:
    		Fitting AsTypeFeatureGenerator...
    			Note: Converting 2 features to boolean dtype as they only contain 2 unique values.
    	Stage 2 Generators:
    		Fitting FillNaFeatureGenerator...
    	Stage 3 Generators:
    		Fitting IdentityFeatureGenerator...
    		Fitting CategoryFeatureGenerator...
    			Fitting CategoryMemoryMinimizeFeatureGenerator...
    		Fitting DatetimeFeatureGenerator...
    	Stage 4 Generators:
    		Fitting DropUniqueFeatureGenerator...
    	Stage 5 Generators:
    		Fitting DropDuplicatesFeatureGenerator...
    	Types of features in original data (raw dtype, special dtypes):
    		('category', []) : 2 | ['season', 'weather']
    		('datetime', []) : 1 | ['datetime']
    		('float', [])    : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])      : 4 | ['holiday', 'workingday', 'humidity', 'hour']
    	Types of features in processed data (raw dtype, special dtypes):
    		('category', [])             : 2 | ['season', 'weather']
    		('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    		('int', [])                  : 2 | ['humidity', 'hour']
    		('int', ['bool'])            : 2 | ['holiday', 'workingday']
    		('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    	0.1s = Fit runtime
    	10 features in original data used to generate 14 features in processed data.
    	Train Data (Processed) Memory Usage: 0.83 MB (0.1% of available memory)
    Data preprocessing and feature engineering runtime = 0.12s ...
    AutoGluon will gauge predictive performance using evaluation metric: 'root_mean_squared_error'
    	This metric's sign has been flipped to adhere to being higher_is_better. The metric score can be multiplied by -1 to get the metric value.
    	To change this, specify the eval_metric parameter of Predictor()
    User-specified model hyperparameters to be fit:
    {
    	'GBM': [{}],
    	'RF': [{}],
    	'XT': [{}],
    	'NN_TORCH': [{}],
    }
    AutoGluon will fit 2 stack levels (L1 to L2) ...
    Fitting 4 L1 models, fit_strategy="sequential" ...
    Hyperparameter tuning model: LightGBM_BAG_L1 ... Tuning model for up to 84.3s of the 562.14s of remaining time.
    Warning: Exception caused LightGBM_BAG_L1 to fail during hyperparameter tuning... Skipping this model.
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.12/site-packages/autogluon/tabular/trainer/abstract_trainer.py", line 2555, in _train_single_full
        hpo_models, hpo_results = model.hyperparameter_tune(
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/abstract/abstract_model.py", line 1891, in hyperparameter_tune
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 289, in _hyperparameter_tune
        return super()._hyperparameter_tune(X=X, y=y, k_fold=k_fold, hpo_executor=hpo_executor, preprocess_kwargs=preprocess_kwargs, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 1603, in _hyperparameter_tune
        hpo_executor.execute(
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/executors.py", line 541, in execute
        scheduler = scheduler_cls(model_trial, search_space=self.search_space, train_fn_kwargs=train_fn_kwargs, **scheduler_params)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/scheduler/seq_scheduler.py", line 97, in __init__
        self.searcher: LocalSearcher = self.get_searcher_(searcher, train_fn, search_space=search_space, **kwargs)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/scheduler/seq_scheduler.py", line 135, in get_searcher_
        searcher = searcher_factory(searcher, **{**scheduler_opts, **_search_options})
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/searcher/searcher_factory.py", line 67, in searcher_factory
        raise AssertionError(f"searcher '{searcher_name}' is not supported")
    AssertionError: searcher 'bayesopt' is not supported
    searcher 'bayesopt' is not supported
    Hyperparameter tuning model: RandomForest_BAG_L1 ... Tuning model for up to 84.3s of the 562.07s of remaining time.
    	No hyperparameter search space specified for RandomForest_BAG_L1. Skipping HPO. Will train one model based on the provided hyperparameters.
    	Warning: Reducing model 'n_estimators' from 300 -> 124 due to low memory. Expected memory usage reduced from 36.25% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    Fitted model: RandomForest_BAG_L1 ...
    	-38.7796	 = Validation score   (-root_mean_squared_error)
    	7.63s	 = Training   runtime
    	0.26s	 = Validation runtime
    Hyperparameter tuning model: ExtraTrees_BAG_L1 ... Tuning model for up to 84.3s of the 554.42s of remaining time.
    	No hyperparameter search space specified for ExtraTrees_BAG_L1. Skipping HPO. Will train one model based on the provided hyperparameters.
    	Warning: Reducing model 'n_estimators' from 300 -> 120 due to low memory. Expected memory usage reduced from 37.27% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    Fitted model: ExtraTrees_BAG_L1 ...
    	-38.7749	 = Validation score   (-root_mean_squared_error)
    	3.18s	 = Training   runtime
    	0.25s	 = Validation runtime
    Hyperparameter tuning model: NeuralNetTorch_BAG_L1 ... Tuning model for up to 84.3s of the 551.22s of remaining time.
    Warning: Exception caused NeuralNetTorch_BAG_L1 to fail during hyperparameter tuning... Skipping this model.
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.12/site-packages/autogluon/tabular/trainer/abstract_trainer.py", line 2555, in _train_single_full
        hpo_models, hpo_results = model.hyperparameter_tune(
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/abstract/abstract_model.py", line 1891, in hyperparameter_tune
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 289, in _hyperparameter_tune
        return super()._hyperparameter_tune(X=X, y=y, k_fold=k_fold, hpo_executor=hpo_executor, preprocess_kwargs=preprocess_kwargs, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 1603, in _hyperparameter_tune
        hpo_executor.execute(
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/executors.py", line 424, in execute
        analysis = run(
                   ^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/ray_hpo.py", line 227, in run
        searcher = _get_searcher(
                   ^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/ray_hpo.py", line 367, in _get_searcher
        assert searcher in SEARCHER_PRESETS, f"{searcher} is not a valid option. Options are {SEARCHER_PRESETS.keys()}"
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: bayesopt is not a valid option. Options are dict_keys(['random', 'bayes'])
    bayesopt is not a valid option. Options are dict_keys(['random', 'bayes'])
    Fitting model: WeightedEnsemble_L2 ... Training model for up to 360.00s of the 550.65s of remaining time.
    	Ensemble Weights: {'RandomForest_BAG_L1': 0.5, 'ExtraTrees_BAG_L1': 0.5}
    	-37.3609	 = Validation score   (-root_mean_squared_error)
    	0.01s	 = Training   runtime
    	0.0s	 = Validation runtime
    Fitting 4 L2 models, fit_strategy="sequential" ...
    Hyperparameter tuning model: LightGBM_BAG_L2 ... Tuning model for up to 123.89s of the 550.62s of remaining time.
    Warning: Exception caused LightGBM_BAG_L2 to fail during hyperparameter tuning... Skipping this model.
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.12/site-packages/autogluon/tabular/trainer/abstract_trainer.py", line 2555, in _train_single_full
        hpo_models, hpo_results = model.hyperparameter_tune(
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/abstract/abstract_model.py", line 1891, in hyperparameter_tune
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 289, in _hyperparameter_tune
        return super()._hyperparameter_tune(X=X, y=y, k_fold=k_fold, hpo_executor=hpo_executor, preprocess_kwargs=preprocess_kwargs, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 1603, in _hyperparameter_tune
        hpo_executor.execute(
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/executors.py", line 541, in execute
        scheduler = scheduler_cls(model_trial, search_space=self.search_space, train_fn_kwargs=train_fn_kwargs, **scheduler_params)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/scheduler/seq_scheduler.py", line 97, in __init__
        self.searcher: LocalSearcher = self.get_searcher_(searcher, train_fn, search_space=search_space, **kwargs)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/scheduler/seq_scheduler.py", line 135, in get_searcher_
        searcher = searcher_factory(searcher, **{**scheduler_opts, **_search_options})
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/searcher/searcher_factory.py", line 67, in searcher_factory
        raise AssertionError(f"searcher '{searcher_name}' is not supported")
    AssertionError: searcher 'bayesopt' is not supported
    searcher 'bayesopt' is not supported
    Hyperparameter tuning model: RandomForest_BAG_L2 ... Tuning model for up to 123.89s of the 550.6s of remaining time.
    	No hyperparameter search space specified for RandomForest_BAG_L2. Skipping HPO. Will train one model based on the provided hyperparameters.
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    Fitted model: RandomForest_BAG_L2 ...
    	-37.7441	 = Validation score   (-root_mean_squared_error)
    	27.02s	 = Training   runtime
    	0.66s	 = Validation runtime
    Hyperparameter tuning model: ExtraTrees_BAG_L2 ... Tuning model for up to 123.89s of the 523.55s of remaining time.
    	No hyperparameter search space specified for ExtraTrees_BAG_L2. Skipping HPO. Will train one model based on the provided hyperparameters.
    	Warning: Reducing model 'n_estimators' from 300 -> 164 due to low memory. Expected memory usage reduced from 27.32% -> 15.0% of available memory...
    /opt/conda/lib/python3.12/site-packages/sklearn/base.py:474: FutureWarning: `BaseEstimator._validate_data` is deprecated in 1.6 and will be removed in 1.7. Use `sklearn.utils.validation.validate_data` instead. This function becomes public and is part of the scikit-learn developer API.
      warnings.warn(
    Fitted model: ExtraTrees_BAG_L2 ...
    	-37.3597	 = Validation score   (-root_mean_squared_error)
    	6.03s	 = Training   runtime
    	0.37s	 = Validation runtime
    Hyperparameter tuning model: NeuralNetTorch_BAG_L2 ... Tuning model for up to 123.89s of the 517.5s of remaining time.
    Warning: Exception caused NeuralNetTorch_BAG_L2 to fail during hyperparameter tuning... Skipping this model.
    Traceback (most recent call last):
      File "/opt/conda/lib/python3.12/site-packages/autogluon/tabular/trainer/abstract_trainer.py", line 2555, in _train_single_full
        hpo_models, hpo_results = model.hyperparameter_tune(
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/abstract/abstract_model.py", line 1891, in hyperparameter_tune
        return self._hyperparameter_tune(hpo_executor=hpo_executor, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/stacker_ensemble_model.py", line 289, in _hyperparameter_tune
        return super()._hyperparameter_tune(X=X, y=y, k_fold=k_fold, hpo_executor=hpo_executor, preprocess_kwargs=preprocess_kwargs, **kwargs)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/models/ensemble/bagged_ensemble_model.py", line 1603, in _hyperparameter_tune
        hpo_executor.execute(
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/executors.py", line 424, in execute
        analysis = run(
                   ^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/ray_hpo.py", line 227, in run
        searcher = _get_searcher(
                   ^^^^^^^^^^^^^^
      File "/opt/conda/lib/python3.12/site-packages/autogluon/core/hpo/ray_hpo.py", line 367, in _get_searcher
        assert searcher in SEARCHER_PRESETS, f"{searcher} is not a valid option. Options are {SEARCHER_PRESETS.keys()}"
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    AssertionError: bayesopt is not a valid option. Options are dict_keys(['random', 'bayes'])
    bayesopt is not a valid option. Options are dict_keys(['random', 'bayes'])
    Fitting model: WeightedEnsemble_L3 ... Training model for up to 360.00s of the 517.47s of remaining time.
    	Ensemble Weights: {'ExtraTrees_BAG_L2': 0.333, 'ExtraTrees_BAG_L1': 0.25, 'RandomForest_BAG_L1': 0.208, 'RandomForest_BAG_L2': 0.208}
    	-36.8389	 = Validation score   (-root_mean_squared_error)
    	0.02s	 = Training   runtime
    	0.0s	 = Validation runtime
    AutoGluon training complete, total runtime = 44.83s ... Best model: WeightedEnsemble_L3 | Estimated inference throughput: 7017.3 rows/s (10886 batch size)
    TabularPredictor saved. To load, use: predictor = TabularPredictor.load("/home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_214930")



```python
predictor_new_hpo.fit_summary()
```

    *** Summary of fit() ***
    Estimated performance of each model:
                     model  score_val              eval_metric  pred_time_val   fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  can_infer  fit_order
    0  WeightedEnsemble_L3 -36.838903  root_mean_squared_error       1.551311  43.879917                0.000646           0.017169            3       True          6
    1    ExtraTrees_BAG_L2 -37.359729  root_mean_squared_error       0.887972  16.839389                0.372243           6.029837            2       True          5
    2  WeightedEnsemble_L2 -37.360899  root_mean_squared_error       0.516306  10.820575                0.000577           0.011024            2       True          3
    3  RandomForest_BAG_L2 -37.744112  root_mean_squared_error       1.178422  37.832911                0.662693          27.023359            2       True          4
    4    ExtraTrees_BAG_L1 -38.774927  root_mean_squared_error       0.252350   3.182878                0.252350           3.182878            1       True          2
    5  RandomForest_BAG_L1 -38.779552  root_mean_squared_error       0.263380   7.626674                0.263380           7.626674            1       True          1
    Number of models trained: 6
    Types of models trained:
    {'StackerEnsembleModel_RF', 'StackerEnsembleModel_XT', 'WeightedEnsembleModel'}
    Bagging used: True  (with 8 folds)
    Multi-layer stack-ensembling used: True  (with 3 levels)
    Feature Metadata (Processed):
    (raw dtype, special dtypes):
    ('category', [])             : 2 | ['season', 'weather']
    ('float', [])                : 3 | ['temp', 'atemp', 'windspeed']
    ('int', [])                  : 2 | ['humidity', 'hour']
    ('int', ['bool'])            : 2 | ['holiday', 'workingday']
    ('int', ['datetime_as_int']) : 5 | ['datetime', 'datetime.year', 'datetime.month', 'datetime.day', 'datetime.dayofweek']
    Plot summary of models saved to file: /home/sagemaker-user/nd009t-c1-intro-to-ml-project-starter/AutogluonModels/ag-20250608_214930/SummaryOfModels.html
    *** End of fit() summary ***





    {'model_types': {'RandomForest_BAG_L1': 'StackerEnsembleModel_RF',
      'ExtraTrees_BAG_L1': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L2': 'WeightedEnsembleModel',
      'RandomForest_BAG_L2': 'StackerEnsembleModel_RF',
      'ExtraTrees_BAG_L2': 'StackerEnsembleModel_XT',
      'WeightedEnsemble_L3': 'WeightedEnsembleModel'},
     'model_performance': {'RandomForest_BAG_L1': -38.77955153433294,
      'ExtraTrees_BAG_L1': -38.77492741356579,
      'WeightedEnsemble_L2': -37.36089885081609,
      'RandomForest_BAG_L2': -37.74411227985954,
      'ExtraTrees_BAG_L2': -37.35972912766415,
      'WeightedEnsemble_L3': -36.838902696696806},
     'model_best': 'WeightedEnsemble_L3',
     'model_paths': {'RandomForest_BAG_L1': ['RandomForest_BAG_L1'],
      'ExtraTrees_BAG_L1': ['ExtraTrees_BAG_L1'],
      'WeightedEnsemble_L2': ['WeightedEnsemble_L2'],
      'RandomForest_BAG_L2': ['RandomForest_BAG_L2'],
      'ExtraTrees_BAG_L2': ['ExtraTrees_BAG_L2'],
      'WeightedEnsemble_L3': ['WeightedEnsemble_L3']},
     'model_fit_times': {'RandomForest_BAG_L1': 7.626673936843872,
      'ExtraTrees_BAG_L1': 3.182877540588379,
      'WeightedEnsemble_L2': 0.011023759841918945,
      'RandomForest_BAG_L2': 27.023359298706055,
      'ExtraTrees_BAG_L2': 6.029837131500244,
      'WeightedEnsemble_L3': 0.017169475555419922},
     'model_pred_times': {'RandomForest_BAG_L1': 0.2633795738220215,
      'ExtraTrees_BAG_L1': 0.252349853515625,
      'WeightedEnsemble_L2': 0.0005769729614257812,
      'RandomForest_BAG_L2': 0.6626925468444824,
      'ExtraTrees_BAG_L2': 0.37224292755126953,
      'WeightedEnsemble_L3': 0.0006463527679443359},
     'num_bag_folds': 8,
     'max_stack_level': 3,
     'model_hyperparams': {'RandomForest_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'ExtraTrees_BAG_L1': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'WeightedEnsemble_L2': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None},
      'RandomForest_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'ExtraTrees_BAG_L2': {'use_orig_features': True,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None,
       'use_child_oof': True},
      'WeightedEnsemble_L3': {'use_orig_features': False,
       'valid_stacker': True,
       'max_base_models': 0,
       'max_base_models_per_type': 'auto',
       'save_bag_folds': True,
       'stratify': 'auto',
       'bin': 'auto',
       'n_bins': None}},
     'leaderboard':                  model  score_val              eval_metric  pred_time_val  \
     0  WeightedEnsemble_L3 -36.838903  root_mean_squared_error       1.551311   
     1    ExtraTrees_BAG_L2 -37.359729  root_mean_squared_error       0.887972   
     2  WeightedEnsemble_L2 -37.360899  root_mean_squared_error       0.516306   
     3  RandomForest_BAG_L2 -37.744112  root_mean_squared_error       1.178422   
     4    ExtraTrees_BAG_L1 -38.774927  root_mean_squared_error       0.252350   
     5  RandomForest_BAG_L1 -38.779552  root_mean_squared_error       0.263380   
     
         fit_time  pred_time_val_marginal  fit_time_marginal  stack_level  \
     0  43.879917                0.000646           0.017169            3   
     1  16.839389                0.372243           6.029837            2   
     2  10.820575                0.000577           0.011024            2   
     3  37.832911                0.662693          27.023359            2   
     4   3.182878                0.252350           3.182878            1   
     5   7.626674                0.263380           7.626674            1   
     
        can_infer  fit_order  
     0       True          6  
     1       True          5  
     2       True          3  
     3       True          4  
     4       True          2  
     5       True          1  }




```python
predictor_new_hpo.leaderboard(silent=True)
```




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
      <th>model</th>
      <th>score_val</th>
      <th>eval_metric</th>
      <th>pred_time_val</th>
      <th>fit_time</th>
      <th>pred_time_val_marginal</th>
      <th>fit_time_marginal</th>
      <th>stack_level</th>
      <th>can_infer</th>
      <th>fit_order</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>WeightedEnsemble_L3</td>
      <td>-36.838903</td>
      <td>root_mean_squared_error</td>
      <td>1.551311</td>
      <td>43.879917</td>
      <td>0.000646</td>
      <td>0.017169</td>
      <td>3</td>
      <td>True</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ExtraTrees_BAG_L2</td>
      <td>-37.359729</td>
      <td>root_mean_squared_error</td>
      <td>0.887972</td>
      <td>16.839389</td>
      <td>0.372243</td>
      <td>6.029837</td>
      <td>2</td>
      <td>True</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WeightedEnsemble_L2</td>
      <td>-37.360899</td>
      <td>root_mean_squared_error</td>
      <td>0.516306</td>
      <td>10.820575</td>
      <td>0.000577</td>
      <td>0.011024</td>
      <td>2</td>
      <td>True</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RandomForest_BAG_L2</td>
      <td>-37.744112</td>
      <td>root_mean_squared_error</td>
      <td>1.178422</td>
      <td>37.832911</td>
      <td>0.662693</td>
      <td>27.023359</td>
      <td>2</td>
      <td>True</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ExtraTrees_BAG_L1</td>
      <td>-38.774927</td>
      <td>root_mean_squared_error</td>
      <td>0.252350</td>
      <td>3.182878</td>
      <td>0.252350</td>
      <td>3.182878</td>
      <td>1</td>
      <td>True</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RandomForest_BAG_L1</td>
      <td>-38.779552</td>
      <td>root_mean_squared_error</td>
      <td>0.263380</td>
      <td>7.626674</td>
      <td>0.263380</td>
      <td>7.626674</td>
      <td>1</td>
      <td>True</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remember to set all negative values to zero
# Remember to set all negative values to zero
predictions_new_hpo = predictor_new_hpo.predict(test)
predictions_new_hpo = predictions_new_hpo.clip(lower=0)
```


```python
submission_new_hpo = submission.copy()
submission_new_hpo["count"] = predictions_new_hpo
submission_new_hpo.to_csv("submission_new_hpo.csv", index=False)
```


```python
!kaggle competitions submit -c bike-sharing-demand -f submission_new_hpo.csv -m "new features with hyperparameters"
```


```python
!kaggle competitions submissions -c bike-sharing-demand | tail -n +1 | head -n 6
```

#### New Score of ` 0.47362`

## Step 7: Write a Report
### Refer to the markdown file for the full report
### Creating plots and table for report


```python
# Taking the top model score from each training run and creating a line plot to show improvement
# You can create these in the notebook and save them to PNG or use some other tool (e.g. google sheets, excel)
fig = pd.DataFrame(
    {
        "model": ["initial", "add_features", "hpo"],
        "score": [53.139808, 30.379192, 1.551311]
    }
).plot(x="model", y="score", marker='o', title="Training RMSE by Model Version", figsize=(8, 6), grid=True)
plt.ylabel("RMSE")
plt.tight_layout()
fig = fig.get_figure()
fig.savefig('model_train_score.png')
```


    
![png](output_70_0.png)
    



```python
# Take the 3 kaggle scores and creating a line plot to show improvement
fig = pd.DataFrame(
    {
        "test_eval": ["initial", "add_features", "hpo"],
        "score": [1.79478, 0.62064, 0.47362]
    }
).plot(
    x="test_eval", 
    y="score", 
    marker='o', 
    title="Kaggle RMSE by Submission", 
    figsize=(8, 6), 
    grid=True
)
plt.ylabel("Kaggle RMSE")
plt.tight_layout()
fig = fig.get_figure()
fig.savefig('model_test_score.png')
```


    
![png](output_71_0.png)
    


### Hyperparameter table


```python
# The 3 hyperparameters we tuned with the kaggle score as the result
pd.DataFrame({
    "model": ["initial", "add_features", "hpo"],
    "hpo1": ["default", "default", "num_trials=10"],
    "hpo2": ["default", "added hour", "included_models=['GBM', 'RF', 'XT', 'NN_TORCH']"],
    "hpo3": ["default", "categorical vars", "search='bayesopt'"],
    "score": [1.79478, 0.62064, 0.47362]
})
```




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
      <th>model</th>
      <th>hpo1</th>
      <th>hpo2</th>
      <th>hpo3</th>
      <th>score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>initial</td>
      <td>default</td>
      <td>default</td>
      <td>default</td>
      <td>1.79478</td>
    </tr>
    <tr>
      <th>1</th>
      <td>add_features</td>
      <td>default</td>
      <td>added hour</td>
      <td>categorical vars</td>
      <td>0.62064</td>
    </tr>
    <tr>
      <th>2</th>
      <td>hpo</td>
      <td>num_trials=10</td>
      <td>included_models=['GBM', 'RF', 'XT', 'NN_TORCH']</td>
      <td>search='bayesopt'</td>
      <td>0.47362</td>
    </tr>
  </tbody>
</table>
</div>


