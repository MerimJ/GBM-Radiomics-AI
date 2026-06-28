import pandas as pd

df = pd.read_csv('data/processed/CFB_GBM_features_extraction_pyradiomics_v03_20260619.tsv', sep='\t')
print('Unique Image values:', df['Image'].unique())
print('Unique Mask values:', df['Mask'].unique())
print('Unique Label name values:', df['Label name'].unique())
print('Unique Sequence values:', df['Sequence'].unique())
print('Unique Temporality values:', df['Temporality'].unique())
print()
print('Rows per patient (mean):', df.groupby('Patient').size().mean())