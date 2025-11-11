"""
Stratify sample from the full dataset to ensure balanced representation
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the full dataset from parquet
df = pd.read_parquet('data/processed/lol_15min_data.parquet')

print(f"Total matches: {len(df)}")
print(f"Blue wins: {df['blue_win'].sum()}")
print(f"Red wins: {len(df) - df['blue_win'].sum()}")
print(f"Win rate: {df['blue_win'].mean():.2%}")

# Stratified sample of 50 matches
sample_size = 50
_, stratified_sample = train_test_split(
    df, 
    test_size=sample_size, 
    stratify=df['blue_win'],
    random_state=42
)

print(f"\nStratified sample: {len(stratified_sample)}")
print(f"Blue wins: {stratified_sample['blue_win'].sum()}")
print(f"Red wins: {len(stratified_sample) - stratified_sample['blue_win'].sum()}")
print(f"Win rate: {stratified_sample['blue_win'].mean():.2%}")

# Save stratified sample
stratified_sample.to_csv('data/processed/lol_15min_stratified_sample.csv', index=False)
print(f"\nStratified sample saved to data/processed/lol_15min_stratified_sample.csv")
