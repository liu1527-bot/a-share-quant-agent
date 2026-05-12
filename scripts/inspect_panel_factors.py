# -*- coding: utf-8 -*-
"""Inspect what factors are in the panel."""
import sys, pickle
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / 'data' / 'cache' / 'factor_panel_hs300.pkl'

with open(PANEL, 'rb') as f:
    data = pickle.load(f)

print("Type:", type(data))
if isinstance(data, dict):
    print("Keys:", list(data.keys())[:30])
    for k, v in list(data.items())[:5]:
        print(f"\n--- {k} ---")
        print(f"  type: {type(v)}")
        if hasattr(v, 'shape'):
            print(f"  shape: {v.shape}")
        if hasattr(v, 'columns'):
            print(f"  columns: {list(v.columns)[:30]}")
        if hasattr(v, 'index'):
            print(f"  index range: {v.index.min()} to {v.index.max()}")
            print(f"  index len: {len(v.index)}")
        if hasattr(v, 'head'):
            print(v.head(3))
elif hasattr(data, 'columns'):
    print("Columns:", list(data.columns))
    print("Shape:", data.shape)
    print("Index:", data.index.names if hasattr(data.index, 'names') else type(data.index))
    print(data.head(5))
