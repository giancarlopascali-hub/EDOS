import olympus
from olympus.datasets import Dataset
from olympus.datasets import list_datasets
import pandas as pd

print("Olympus version:", olympus.__version__ if hasattr(olympus, '__version__') else 'unknown')
print("Available datasets:", list_datasets())

try:
    d1 = Dataset(kind='suzuki_i')
    print("Suzuki I found")
    print(d1.data.head())
except Exception as e:
    print(f"Suzuki I error: {e}")

try:
    d4 = Dataset(kind='suzuki_iv')
    print("Suzuki IV found")
    print(d4.data.head())
except Exception as e:
    print(f"Suzuki IV error: {e}")
