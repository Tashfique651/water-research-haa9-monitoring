"""Linux fallback used for the recorded test run; standard use is pytest + PyArrow."""
from pathlib import Path
import os,sys
import pandas as pd
import pytest
from flat_parquet_reader import read_parquet
ROOT=Path(__file__).resolve().parents[2]
pd.read_parquet=lambda path,columns=None,**kwargs: read_parquet(path,columns=columns)
os.chdir(ROOT)
raise SystemExit(pytest.main(['tests','-q','--disable-warnings']))
