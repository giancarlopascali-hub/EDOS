from olymp.datasets import Dataset
import pandas as pd

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

# List all available datasets
from olymp.datasets import list_datasets
print("Available datasets:", list_datasets())
