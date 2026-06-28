import pandas as pd
import os

files = [f for f in os.listdir('data/processed') if f.endswith('.tsv')]
for f in files:
    df = pd.read_csv(f'data/processed/{f}', sep='\t')
    print(f'=== {f} ===')
    print(f'Shape: {df.shape}')
    print(f'Columns: {list(df.columns)}')
    print()