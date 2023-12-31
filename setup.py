from setuptools import setup
from setuptools import setup, find_packages
setup(
    name='ml_aut',
    version='0.1',
    packages=find_packages(),
    install_requires=['absl-py>=2.0.0', 'aiosignal>=1.3.1', 'alembic>=1.13.0',
                      'altair>=5.2.0', 'appdirs>=1.4.4', 'astunparse>=1.6.3',
                      'attrs>=23.1.0', 'beautifulsoup4>=4.12.2', 'blinker>=1.7.0',
                      'cachetools>=5.3.2', 'catboost>=1.2.2', 'certifi>=2023.11.17',
                      'charset-normalizer>=3.3.2', 'click>=8.1.7', 'cloudpickle>=3.0.0',
                      'colorlog>=6.8.0', 'contourpy>=1.2.0', 'cycler>=0.12.1', 'dask>=2023.12.0',
                      'distributed>=2023.12.0', 'feature-engine>=1.6.2', 'filelock>=3.13.1',
                      'flatbuffers>=23.5.26', 'fonttools>=4.46.0', 'frozendict>=2.3.10', 'frozenlist>=1.4.0',
                      'fsspec>=2023.12.1', 'gast>=0.5.4', 'gitdb>=4.0.11', 'GitPython>=3.1.40',
                      'google-auth>=2.25.2', 'google-auth-oauthlib>=1.1.0', 'google-pasta>=0.2.0',
                      'graphviz>=0.20.1', 'greenlet>=3.0.2', 'grpcio>=1.60.0', 'h5py>=3.10.0',
                      'html5lib>=1.1', 'idna>=3.6', 'imbalanced-learn>=0.11.0', 'imblearn>=0.0',
                      'importlib-metadata>=6.11.0', 'Jinja2>=3.1.2', 'joblib>=1.3.2', 'jsonschema>=4.20.0',
                      'jsonschema-specifications>=2023.11.2', 'keras>=2.15.0', 'keras-self-attention>=0.51.0',
                      'kiwisolver>=1.4.5', 'libclang>=16.0.6', 'lightgbm>=4.1.0', 'llvmlite>=0.41.1', 'locket>=1.0.0',
                      'lxml>=4.9.3', 'Mako>=1.3.0', 'Markdown>=3.5.1', 'markdown-it-py>=3.0.0', 'MarkupSafe>=2.1.3',
                      'matplotlib>=3.8.2', 'mdurl>=0.1.2', 'ml-dtypes>=0.2.0', 'mpmath>=1.3.0', 'msgpack>=1.0.7',
                      'multitasking>=0.0.11', 'networkx>=3.2.1', 'numba>=0.58.1', 'numpy>=1.26.2', 'nvidia-cublas-cu12>=12.1.3.1',
                      'nvidia-cuda-cupti-cu12>=12.1.105', 'nvidia-cuda-nvrtc-cu12>=12.1.105', 'nvidia-cuda-runtime-cu12>=12.1.105',
                      'nvidia-cudnn-cu12>=8.9.2.26', 'nvidia-cufft-cu12>=11.0.2.54', 'nvidia-curand-cu12>=10.3.2.106',
                      'nvidia-cusolver-cu12>=11.4.5.107', 'nvidia-cusparse-cu12>=12.1.0.106', 'nvidia-nccl-cu12>=2.18.1',
                      'nvidia-nvjitlink-cu12>=12.3.101', 'nvidia-nvtx-cu12>=12.1.105', 'oauthlib>=3.2.2', 'opt-einsum>=3.3.0',
                      'optuna>=3.4.0', 'packaging>=23.2', 'pandas>=2.1.4', 'partd>=1.4.1', 'patsy>=0.5.4', 'peewee>=3.17.0',
                      'Pillow>=10.1.0', 'plotly>=5.18.0', 'polars>=0.19.19', 'protobuf>=4.23.4', 'psutil>=5.9.6', 'pyarrow>=14.0.1',
                      'pyasn1>=0.5.1', 'pyasn1-modules>=0.3.0', 'pydeck>=0.8.1b0', 'Pygments>=2.17.2', 'pynndescent>=0.5.11',
                      'pyparsing>=3.1.1', 'python-dateutil>=2.8.2', 'pytz>=2023.3.post1', 'PyYAML>=6.0.1', 'ray>=2.8.1', 'referencing>=0.32.0',
                      'requests>=2.31.0', 'requests-oauthlib>=1.3.1', 'rich>=13.7.0', 'rpds-py>=0.13.2', 'rsa>=4.9', 'scikit-base>=0.6.1',
                      'scikit-learn>=1.3.2', 'scipy>=1.11.4', 'seaborn>=0.13.0', 'six>=1.16.0', 'sktime>=0.24.1', 'smmap>=5.0.1',
                      'sortedcontainers>=2.4.0', 'soupsieve>=2.5', 'SQLAlchemy>=2.0.23', 'statsmodels>=0.14.0', 'streamlit>=1.29.0',
                      'stumpy>=1.12.0', 'sympy>=1.12', 'tblib>=3.0.0', 'tenacity>=8.2.3', 'tensorboard>=2.15.1', 'tensorboard-data-server>=0.7.2',
                      'tensorflow>=2.15.0.post1', 'tensorflow-estimator>=2.15.0', 'tensorflow-io-gcs-filesystem>=0.34.0', 'termcolor>=2.4.0',
                      'threadpoolctl>=3.2.0', 'toml>=0.10.2', 'toolz>=0.12.0', 'torch>=2.1.1', 'tornado>=6.4', 'tqdm>=4.66.1', 'triton>=2.1.0',
                      'tsfresh>=0.20.1', 'typing_extensions>=4.8.0', 'tzdata>=2023.3', 'tzlocal>=5.2', 'umap-learn>=0.5.5', 'urllib3>=2.1.0',
                      'validators>=0.22.0', 'watchdog>=3.0.0', 'webencodings>=0.5.1', 'Werkzeug>=3.0.1', 'wrapt>=1.14.1', 'xgboost>=2.0.2',
                      'yahoofinance>=0.0.2', 'yfinance>=0.2.33', 'zict>=3.0.0', 'zipp>=3.17.0'],
    url='https://github.com/tempathack/ml_aut.git',
    license='',
    author='tempathack',
    author_email='tobias_schmidbauer@gmx.de',
    description='The ML Aut framework is a simple framework that allows you to train and test models in Python combining the functions of sktime and sklearn.',
)

