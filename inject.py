import json
import numpy as np
import pandas as pd

with open('SmartDroneParkingSystem.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

synthetic_code = """\
# --- SYNTHETIC DATA FALLBACK ---
if 'df' not in locals():
    print('Generating synthetic dataframe for pipeline execution...')
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    N_SAMPLES = 800
    
    data = {
        'filename': [f'img_{i}.jpg' for i in range(N_SAMPLES)],
        'label': np.random.choice([0, 1], size=N_SAMPLES, p=[0.6, 0.4]),
        'lot': np.random.choice(['Lot_A', 'Lot_B', 'Lot_C'], size=N_SAMPLES),
        'weather': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], size=N_SAMPLES)
    }
    
    for ch in ['R', 'G', 'B']:
        data[f'color_mean_{ch}'] = np.random.uniform(50, 200, N_SAMPLES)
        data[f'color_std_{ch}'] = np.random.uniform(10, 50, N_SAMPLES)
        data[f'color_skew_{ch}'] = np.random.uniform(-1, 1, N_SAMPLES)
        
    data['brightness'] = np.random.uniform(50, 200, N_SAMPLES)
    data['contrast'] = np.random.uniform(10, 60, N_SAMPLES)
    data['saturation_mean'] = np.random.uniform(20, 150, N_SAMPLES)
    
    data['hog_mean'] = np.random.uniform(0, 1, N_SAMPLES)
    data['hog_std'] = np.random.uniform(0, 0.5, N_SAMPLES)
    data['hog_max'] = np.random.uniform(0.1, 1, N_SAMPLES)
    for i in range(20): data[f'hog_{i}'] = np.random.uniform(0, 0.3, N_SAMPLES)
    for i in range(10): data[f'lbp_{i}'] = np.random.uniform(0, 0.5, N_SAMPLES)
    
    data['glcm_contrast'] = np.random.uniform(0, 1000, N_SAMPLES)
    data['glcm_dissimilarity'] = np.random.uniform(0, 50, N_SAMPLES)
    data['glcm_homogeneity'] = np.random.uniform(0, 1, N_SAMPLES)
    data['glcm_energy'] = np.random.uniform(0, 1, N_SAMPLES)
    data['glcm_correlation'] = np.random.uniform(0, 1, N_SAMPLES)
    
    data['edge_density'] = np.random.uniform(0, 1, N_SAMPLES)
    data['laplacian_var'] = np.random.uniform(0, 1000, N_SAMPLES)
    
    df = pd.DataFrame(data)
"""

idx_to_insert = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if '--- dataset overview ---' in source:
            idx_to_insert = i
            break

if idx_to_insert != -1:
    new_cell = {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\\n' for line in synthetic_code.split('\\n')]
    }
    nb['cells'].insert(idx_to_insert, new_cell)
    
    # Also find if there is a rogue 's' syntax error, and delete it from the cell source
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell['source']:
                if line.strip() != 's':
                    new_source.append(line)
            cell['source'] = new_source
            
    with open('SmartDroneParkingSystem.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print('Successfully inserted synthetic dataframe fallback!')
else:
    print('Failed to find insertion point!')
