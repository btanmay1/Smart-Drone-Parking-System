# CELL
!pip install -q scikit-learn pandas numpy matplotlib seaborn
!pip install -q opencv-python-headless Pillow scikit-image xgboost lightgbm
!pip install -q kaggle tqdm joblib imbalanced-learn

# tried pycaret once — crashed with a version conflict, gave up (see "what didn't work")
# CELL
import os, warnings, time, joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import cv2
from PIL import Image
from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops
from skimage.color import rgb2gray

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             f1_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, VotingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

plt.style.use('seaborn-v0_8-whitegrid')
PALETTE  = ['#2ECC71', '#E74C3C']
CMAP     = 'RdYlGn_r'
FIG_DPI  = 120

print('imports done')
print(f'numpy {np.__version__} | pandas {pd.__version__} | opencv {cv2.__version__}')
# CELL

DATA_DIR       = Path('./pklot_data')
USE_REAL_IMAGES = DATA_DIR.exists() and any(DATA_DIR.rglob('*.jpg'))

print(f'data dir exists: {DATA_DIR.exists()}')
print(f'real images found: {USE_REAL_IMAGES}')
if not USE_REAL_IMAGES:
    print('running in demo mode with synthetic features')
    print('place PKLot folder at ./pklot_data/PKLot/ to use real data')
# CELL
IMG_SIZE = (64, 64)  

def extract_features(img_path: str) -> dict:
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    feats = {}

    
    for i, ch in enumerate(['R', 'G', 'B']):
        channel = img_rgb[:, :, i]
        feats[f'color_mean_{ch}']  = channel.mean()
        feats[f'color_std_{ch}']   = channel.std()
        feats[f'color_skew_{ch}']  = float(pd.Series(channel.flatten()).skew())

    feats['brightness']      = gray.mean()
    feats['contrast']        = gray.std()
    feats['saturation_mean'] = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)[:,:,1].mean()

   
    hog_feats = hog(gray, orientations=8, pixels_per_cell=(8,8),
                    cells_per_block=(2,2), feature_vector=True)
    feats['hog_mean'] = hog_feats.mean()
    feats['hog_std']  = hog_feats.std()
    feats['hog_max']  = hog_feats.max()
    for j, v in enumerate(hog_feats[:20]):
        feats[f'hog_{j}'] = v

    
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
    for j, v in enumerate(lbp_hist):
        feats[f'lbp_{j}'] = v

    
    glcm = graycomatrix((gray // 16).astype(np.uint8), distances=[1],
                        angles=[0], levels=16, symmetric=True, normed=True)
    feats['glcm_contrast']     = graycoprops(glcm, 'contrast')[0,0]
    feats['glcm_dissimilarity']= graycoprops(glcm, 'dissimilarity')[0,0]
    feats['glcm_homogeneity']  = graycoprops(glcm, 'homogeneity')[0,0]
    feats['glcm_energy']       = graycoprops(glcm, 'energy')[0,0]
    feats['glcm_correlation']  = graycoprops(glcm, 'correlation')[0,0]

    
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)
    feats['edge_density'] = edges.mean() / 255.0
    feats['laplacian_var']= cv2.Laplacian(gray, cv2.CV_64F).var()

    return feats
# CELL

print('--- dataset overview ---')
print(f'Shape : {df.shape[0]:,} rows × {df.shape[1]} columns')
print(f'Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
print(f'\nClass distribution:')
vc = df['label'].value_counts()
print(f'  Occupied (1): {vc[1]:,}  ({vc[1]/len(df)*100:.1f}%)')
print(f'  Empty    (0): {vc[0]:,}  ({vc[0]/len(df)*100:.1f}%)')
print(f'\nLots   : {df["lot"].value_counts().to_dict()}')
print(f'Weather: {df["weather"].value_counts().to_dict()}')
df.head(3)
# CELL
df_raw = df.copy()

print(' DATA AUDIT')
print(f'Shape: {df_raw.shape}')
print(f'Memory: {df_raw.memory_usage(deep=True).sum() / 1e6:.2f} MB')

num_cols_raw = df_raw.select_dtypes(include=np.number).columns.tolist()
cat_cols_raw = df_raw.select_dtypes(include='object').columns.tolist()
print(f'Numeric cols: {len(num_cols_raw)} | Object cols: {cat_cols_raw}')

s
null_counts = df_raw.isnull().sum()
missing_df  = pd.DataFrame({'count': null_counts, 'pct_%': null_counts/len(df_raw)*100})
missing_df  = missing_df[missing_df['count'] > 0].sort_values('count', ascending=False)
print(f'\nMissing values:')
if missing_df.empty:
    print('  none found')
else:
    print(missing_df.head(20).to_string())


inf_counts = {c: np.isinf(df_raw[c]).sum() for c in num_cols_raw if np.isinf(df_raw[c]).sum() > 0}
print(f'\nInfinite values: {sum(inf_counts.values())} total')
if inf_counts:
    print('  Columns:', inf_counts)


print(f'\nDuplicate rows: {df_raw.duplicated().sum()}')


zero_var = [c for c in num_cols_raw if df_raw[c].nunique() == 1]
low_var  = [c for c in num_cols_raw if df_raw[c].std() < 1e-6 and c not in zero_var]
print(f'Zero-variance features: {len(zero_var)} {zero_var}')
print(f'Near-zero variance    : {len(low_var)} {low_var}')


vc_raw     = df_raw['label'].value_counts()
imb_ratio  = vc_raw.max() / vc_raw.min()
print(f'\nClass balance:')
print(f'  Occupied (1): {vc_raw.get(1,0):,}  ({vc_raw.get(1,0)/len(df_raw)*100:.1f}%)')
print(f'  Empty    (0): {vc_raw.get(0,0):,}  ({vc_raw.get(0,0)/len(df_raw)*100:.1f}%)')
print(f'  Imbalance ratio: {imb_ratio:.2f}x')
print('audit done')
# CELL
from sklearn.impute import SimpleImputer, KNNImputer

df_missing = df_raw.copy()


rng_miss = np.random.default_rng(99)
injection_plan = {
    'glcm_correlation': 0.03,
    'laplacian_var':    0.04,
    'saturation_mean':  0.07,
    'hog_mean':         0.02,
    'color_skew_R':     0.15,
}
for col, frac in injection_plan.items():
    idx = rng_miss.choice(len(df_missing), int(frac * len(df_missing)), replace=False)
    df_missing.loc[idx, col] = np.nan

print('Injected missing values:')
for col, frac in injection_plan.items():
    n = df_missing[col].isnull().sum()
    print(f'  {col:<25}: {n:>5} NaNs ({frac*100:.0f}%)')

num_feat_cols = [c for c in df_missing.select_dtypes(include=np.number).columns if c != 'label']
miss_rates    = df_missing[num_feat_cols].isnull().mean()

drop_cols_missing  = miss_rates[miss_rates > 0.30].index.tolist()
knn_impute_cols    = miss_rates[(miss_rates > 0.05) & (miss_rates <= 0.30)].index.tolist()
simple_impute_cols = miss_rates[(miss_rates > 0) & (miss_rates <= 0.05)].index.tolist()

print(f'\nImputation plan:')
print(f'  DROP (>30%)      : {drop_cols_missing}')
print(f'  KNN impute (5-30%): {knn_impute_cols}')
print(f'  Simple (<5%)     : {simple_impute_cols}')


flag_cols = miss_rates[miss_rates > 0.01].index.tolist()
for col in flag_cols:
    df_missing[f'{col}__missing'] = df_missing[col].isnull().astype(int)

if drop_cols_missing:
    df_missing.drop(columns=drop_cols_missing, inplace=True)

def choose_strategy(series):
    return 'median' if abs(series.dropna().skew()) > 1.0 else 'mean'

df_imputed = df_missing.copy()

for col in simple_impute_cols:
    if col not in df_imputed.columns: continue
    fill_val = df_imputed[col].median() if choose_strategy(df_imputed[col]) == 'median' else df_imputed[col].mean()
    df_imputed[col].fillna(fill_val, inplace=True)

if knn_impute_cols:
    knn_imp = KNNImputer(n_neighbors=5, weights='distance')
    df_imputed[knn_impute_cols] = knn_imp.fit_transform(df_imputed[knn_impute_cols])

remaining = df_imputed[num_feat_cols].isnull().sum().sum()
print(f'\nRemaining NaNs after imputation: {remaining}')
print(f'Dataset shape: {df_imputed.shape}')
# CELL
from scipy import stats as sp_stats

df_iqr = df_imputed.copy()
feat_num = [c for c in df_iqr.select_dtypes(include=np.number).columns
            if c not in ['label'] and '__missing' not in c]

iqr_report = []
iqr_bounds = {}

for col in feat_num:
    q1  = df_iqr[col].quantile(0.25)
    q3  = df_iqr[col].quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - 1.5 * iqr
    hi  = q3 + 1.5 * iqr
    n_lo, n_hi = (df_iqr[col] < lo).sum(), (df_iqr[col] > hi).sum()
    iqr_bounds[col] = (lo, hi)
    iqr_report.append({'feature': col, 'total_outliers': n_lo + n_hi,
                        'outlier_%': round((n_lo + n_hi) / len(df_iqr) * 100, 2),
                        'lower_fence': round(lo, 4), 'upper_fence': round(hi, 4)})

iqr_df = pd.DataFrame(iqr_report).sort_values('total_outliers', ascending=False)
print('Top 15 features by outlier count (IQR):')
print(iqr_df.head(15)[['feature','total_outliers','outlier_%','lower_fence','upper_fence']].to_string(index=False))


df_winsor = df_iqr.copy()
for col in feat_num:
    lo, hi = iqr_bounds[col]
    df_winsor[col] = df_winsor[col].clip(lower=lo, upper=hi)

print('\nWinsorization done. Extreme tails capped at IQR fences.')
# CELL
from sklearn.preprocessing import OneHotEncoder

df_encoded = df_winsor.copy()
cat_cols = [c for c in df_encoded.select_dtypes(include='object').columns if c != 'filename']
print(f'Categorical cols to encode: {cat_cols}')

ohe_full = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')

for col in cat_cols:
    
    ohe_arr   = ohe_full.fit_transform(df_encoded[[col]])
    ohe_names = [f'{col}__{cat}' for cat in ohe_full.categories_[0]]
    df_encoded = pd.concat([df_encoded, pd.DataFrame(ohe_arr, columns=ohe_names, index=df_encoded.index)], axis=1)
    
    df_encoded[f'{col}__label'] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
    
    freq_map = df_encoded[col].value_counts(normalize=True).to_dict()
    df_encoded[f'{col}__freq'] = df_encoded[col].map(freq_map)
    print(f'  {col}: OHE={ohe_names} | also label + freq')

cols_to_drop = cat_cols + [f for f in ['filename'] if f in df_encoded.columns]
df_encoded.drop(columns=cols_to_drop, errors='ignore', inplace=True)

print(f'\nShape after encoding: {df_encoded.shape}')
# CELL
y_full = df_encoded['label'].astype(int)
X_full = df_encoded.drop(columns=['label'], errors='ignore').select_dtypes(include=np.number)

X_full.replace([np.inf, -np.inf], np.nan, inplace=True)
X_full.fillna(X_full.median(), inplace=True)

print(f'Feature matrix: {X_full.shape}')
print(f'Positive rate : {y_full.mean():.3f}')


X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_full, y_full, test_size=0.15, random_state=42, stratify=y_full)


X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.1765, random_state=42, stratify=y_trainval)

print(f'\nTrain : {len(X_train):,} ({len(X_train)/len(X_full)*100:.0f}%)')
print(f'Val   : {len(X_val):,}  ({len(X_val)/len(X_full)*100:.0f}%)')
print(f'Test  : {len(X_test):,}  ({len(X_test)/len(X_full)*100:.0f}%)')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print('\nSplits ready. StratifiedKFold (k=5) configured.')
# CELL
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = RobustScaler(quantile_range=(25, 75))
scaler.fit(X_train)  

X_train_sc = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_sc   = pd.DataFrame(scaler.transform(X_val),   columns=X_val.columns,   index=X_val.index)
X_test_sc  = pd.DataFrame(scaler.transform(X_test),  columns=X_test.columns,  index=X_test.index)

print('RobustScaler applied (fit on train only):')
print(f'  X_train_sc : {X_train_sc.shape}  | mean≈{X_train_sc.values.mean():.4f}')
print(f'  X_val_sc   : {X_val_sc.shape}    | mean≈{X_val_sc.values.mean():.4f}')
print(f'  X_test_sc  : {X_test_sc.shape}   | mean≈{X_test_sc.values.mean():.4f}')
print('  (val/test means slightly off zero = expected, not leakage)')
# CELL
!pip install -q imbalanced-learn

from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler

train_counts  = Counter(y_train)
imb_ratio_train = max(train_counts.values()) / min(train_counts.values())
print(f'Training set class balance:')
print(f'  Occupied (1): {train_counts[1]:,}')
print(f'  Empty    (0): {train_counts[0]:,}')
print(f'  Imbalance ratio: {imb_ratio_train:.2f}x')

print('\nApplying resampling strategies...')
smote       = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
bsmote      = BorderlineSMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
smote_tomek = SMOTETomek(sampling_strategy='auto', random_state=42)
rus         = RandomUnderSampler(sampling_strategy='auto', random_state=42)

X_train_smote,  y_train_smote  = smote.fit_resample(X_train_sc, y_train)
X_train_bsmote, y_train_bsmote = bsmote.fit_resample(X_train_sc, y_train)
X_train_st,     y_train_st     = smote_tomek.fit_resample(X_train_sc, y_train)
X_train_rus,    y_train_rus    = rus.fit_resample(X_train_sc, y_train)

print(f'\n{"Strategy":<22} {"N_samples":>10} {"Occ":>8} {"Emp":>8}')
for name, ys in [("Original", y_train), ("SMOTE", y_train_smote),
                  ("BorderlineSMOTE", y_train_bsmote), ("SMOTE-Tomek", y_train_st),
                  ("RandomUndersample", y_train_rus)]:
    c = Counter(ys)
    print(f'{name:<22} {len(ys):>10,} {c[1]:>8,} {c[0]:>8,}')
# CELL

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Target Variable — Parking Occupancy', fontsize=14, fontweight='bold')

vc = df_clean['label'].value_counts().sort_index()

wedges, texts, autotexts = axes[0].pie(
    vc.values, labels=['Empty (0)', 'Occupied (1)'], colors=PALETTE[::-1],
    autopct='%1.2f%%', startangle=90, pctdistance=0.80,
    wedgeprops={'edgecolor': 'white', 'linewidth': 2.5, 'width': 0.55})
for at in autotexts:
    at.set_fontsize(10); at.set_fontweight('bold')
axes[0].set_title('Class Split', fontweight='bold')
axes[0].text(0, 0, f'{len(df_clean):,}\nsamples', ha='center', va='center',
             fontsize=11, fontweight='bold')


lot_occ = df_clean.groupby(['lot', 'label']).size().unstack(fill_value=0)
lot_occ.columns = ['Empty', 'Occupied']
lot_occ.plot(kind='bar', ax=axes[1], color=PALETTE, width=0.65, edgecolor='white')
axes[1].set_title('By Parking Lot', fontweight='bold')
axes[1].set_xlabel('Lot'); axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=0)


wx_occ = df_clean.groupby(['weather', 'label']).size().unstack(fill_value=0)
wx_occ.columns = ['Empty', 'Occupied']
wx_occ.plot(kind='bar', ax=axes[2], color=PALETTE, width=0.65, edgecolor='white')
axes[2].set_title('By Weather', fontweight='bold')
axes[2].set_xlabel('Weather'); axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=25)

plt.tight_layout()
plt.savefig('eda_target.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL

num_cols_with_label = [c for c in df_clean.select_dtypes(include='number').columns]
corr_matrix = df_clean[num_cols_with_label].corr()

fig = plt.figure(figsize=(20, 12))
gs  = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.4, wspace=0.35)

ax_heat = fig.add_subplot(gs[0, :])
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.3, annot_kws={'size': 7},
            cbar_kws={'shrink': 0.7}, ax=ax_heat, vmin=-1, vmax=1)
ax_heat.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
ax_heat.tick_params(axis='x', rotation=45, labelsize=8)
ax_heat.tick_params(axis='y', rotation=0, labelsize=8)

ax_bar = fig.add_subplot(gs[1, 0])
target_corr = corr_matrix['label'].drop('label').sort_values(ascending=False)
colors_bar  = [PALETTE[1] if v >= 0 else PALETTE[0] for v in target_corr.values]
ax_bar.barh(target_corr.index, target_corr.values, color=colors_bar, edgecolor='white')
ax_bar.axvline(0, color='black', linewidth=0.8)
ax_bar.set_title('Correlation with Target', fontweight='bold', fontsize=10)
ax_bar.set_xlabel('Pearson r')
ax_bar.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()

print('Top positive correlations with label:')
print(target_corr[target_corr > 0].head(8))
# CELL

violin_features = ['brightness', 'contrast', 'saturation_mean',
                   'edge_density', 'laplacian_var', 'hog_mean',
                   'glcm_energy', 'glcm_contrast', 'glcm_homogeneity',
                   'color_mean_R', 'color_std_R']
violin_features = [f for f in violin_features if f in df_clean.columns]

n_r = int(np.ceil(len(violin_features) / 4))
fig, axes = plt.subplots(n_r, 4, figsize=(18, n_r * 4))
fig.suptitle('Feature Distributions by Class (violin plots)', fontsize=13, fontweight='bold')
axes = axes.flatten()

df_vln = df_clean.copy()
df_vln['Class'] = df_vln['label'].map({0: 'Empty', 1: 'Occupied'})

for i, feat in enumerate(violin_features):
    sns.violinplot(data=df_vln, x='Class', y=feat,
                   palette={'Empty': PALETTE[0], 'Occupied': PALETTE[1]},
                   inner='quartile', cut=0.5, linewidth=1.2, ax=axes[i])
    for j, cls in enumerate(['Empty', 'Occupied']):
        mean_val = df_vln[df_vln['Class'] == cls][feat].mean()
        axes[i].scatter(j, mean_val, marker='D', s=35, color='white',
                        zorder=5, edgecolors='black', linewidths=0.8)
    axes[i].set_title(feat.replace('_', ' ').title(), fontsize=9, fontweight='bold')
    axes[i].set_xlabel('')

for j in range(len(violin_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('eda_violins.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL

feat_cols_pca = [c for c in df_clean.select_dtypes(include='number').columns if c != 'label']
sample_pca    = df_clean.sample(min(3000, len(df_clean)), random_state=42)
X_pca_in      = StandardScaler().fit_transform(sample_pca[feat_cols_pca])

pca_full = PCA(random_state=42).fit(X_pca_in)
cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
n95      = int(np.argmax(cumvar >= 0.95)) + 1
n80      = int(np.argmax(cumvar >= 0.80)) + 1

pca2d = PCA(n_components=2, random_state=42)
X_2d  = pca2d.fit_transform(X_pca_in)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('PCA Projection', fontsize=13, fontweight='bold')

for lbl, color, name in [(0, PALETTE[0], 'Empty'), (1, PALETTE[1], 'Occupied')]:
    mask = sample_pca['label'].values == lbl
    axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, alpha=0.35, s=14,
                    label=name, edgecolors='none')
axes[0].set_xlabel(f'PC1  ({pca2d.explained_variance_ratio_[0]*100:.1f}% var)')
axes[0].set_ylabel(f'PC2  ({pca2d.explained_variance_ratio_[1]*100:.1f}% var)')
axes[0].set_title('2D PCA Projection', fontweight='bold')
axes[0].legend()

axes[1].plot(range(1, len(cumvar)+1), cumvar * 100, 'o-', color='#3498DB', ms=3, lw=1.5)
axes[1].axhline(95, color='red',    ls='--', lw=1.5, label=f'95% → {n95} PCs')
axes[1].axhline(80, color='orange', ls='--', lw=1.5, label=f'80% → {n80} PCs')
axes[1].set_xlabel('Number of PCs')
axes[1].set_ylabel('Cumulative Variance (%)')
axes[1].set_title('Explained Variance', fontweight='bold')
axes[1].legend()

plt.tight_layout()
plt.savefig('eda_pca.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print(f'Components for 95% variance: {n95} | 80%: {n80}')
# CELL
df_eng = df_clean.copy()


df_eng['lot_enc']     = LabelEncoder().fit_transform(df_eng['lot'])
df_eng['weather_enc'] = LabelEncoder().fit_transform(df_eng['weather'])
weather_ohe = pd.get_dummies(df_eng['weather'], prefix='wx').astype(int)
df_eng = pd.concat([df_eng, weather_ohe], axis=1)

eps = 1e-6


df_eng['RG_ratio'] = df_eng['color_mean_R'] / (df_eng['color_mean_G'] + eps)
df_eng['RB_ratio'] = df_eng['color_mean_R'] / (df_eng['color_mean_B'] + eps)
df_eng['GB_ratio'] = df_eng['color_mean_G'] / (df_eng['color_mean_B'] + eps)


df_eng['color_uniformity'] = 1 / (df_eng[['color_std_R','color_std_G','color_std_B']].mean(axis=1) + eps)


df_eng['bright_contrast'] = df_eng['brightness'] * df_eng['contrast']
df_eng['texture_score']   = df_eng['glcm_contrast'] * df_eng['edge_density'] + df_eng['laplacian_var'] / 1000
df_eng['hog_energy']      = df_eng['hog_mean'] ** 2 + df_eng['hog_std'] ** 2

lbp_cols = [f'lbp_{j}' for j in range(10)]
lbp_arr  = df_eng[lbp_cols].values + eps
df_eng['lbp_entropy'] = -(lbp_arr * np.log(lbp_arr)).sum(axis=1)


SLOTS_PER_ROW = 10
df_eng['slot_idx'] = df_eng.groupby('lot').cumcount()
df_eng['slot_row'] = df_eng['slot_idx'] // SLOTS_PER_ROW
df_eng['slot_col'] = df_eng['slot_idx'] %  SLOTS_PER_ROW


df_eng['dist_from_entrance'] = np.sqrt(df_eng['slot_row']**2 + df_eng['slot_col']**2)
max_dist = df_eng['dist_from_entrance'].max()
df_eng['dist_from_entrance_norm'] = df_eng['dist_from_entrance'] / (max_dist + eps)


row_stats = df_eng.groupby(['lot', 'slot_row'])['label'].agg(
    occupied_in_row='sum', total_in_row='count').reset_index()
row_stats['row_occupancy_rate'] = row_stats['occupied_in_row'] / row_stats['total_in_row']
df_eng = df_eng.merge(row_stats[['lot', 'slot_row', 'row_occupancy_rate']],
                      on=['lot', 'slot_row'], how='left')


slot_lookup = df_eng.set_index(['lot', 'slot_row', 'slot_col'])['label'].to_dict()
def get_nbr(lot, row, col, dc):
    nc = col + dc
    if nc < 0 or nc >= SLOTS_PER_ROW: return -1
    return slot_lookup.get((lot, row, nc), -1)

df_eng['left_slot_occupied']  = df_eng.apply(lambda r: get_nbr(r['lot'], r['slot_row'], r['slot_col'], -1), axis=1)
df_eng['right_slot_occupied'] = df_eng.apply(lambda r: get_nbr(r['lot'], r['slot_row'], r['slot_col'], +1), axis=1)
df_eng['adjacent_occupancy_count'] = df_eng['left_slot_occupied'].clip(lower=0) + df_eng['right_slot_occupied'].clip(lower=0)


wx_stats = df_eng.groupby('weather')['brightness'].agg(['mean','std']).reset_index()
wx_stats.columns = ['weather', 'wx_bright_mean', 'wx_bright_std']
df_eng = df_eng.merge(wx_stats, on='weather', how='left')
df_eng['brightness_zscore'] = (df_eng['brightness'] - df_eng['wx_bright_mean']) / (df_eng['wx_bright_std'] + eps)
df_eng['relative_brightness'] = df_eng['brightness'] / (df_eng.groupby('lot')['brightness'].transform('mean') + eps)
q25_by_lot = df_eng.groupby('lot')['brightness'].transform(lambda x: x.quantile(0.25))
df_eng['is_dark_slot'] = (df_eng['brightness'] < q25_by_lot).astype(int)
df_eng.drop(columns=['wx_bright_mean','wx_bright_std'], inplace=True)


lot_occ2 = df_eng.groupby('lot')['label'].mean().reset_index()
lot_occ2.columns = ['lot', 'lot_occupancy_ratio']
df_eng = df_eng.merge(lot_occ2, on='lot', how='left')
global_mean = df_eng['lot_occupancy_ratio'].mean()
global_std  = df_eng['lot_occupancy_ratio'].std()
df_eng['lot_occupancy_zscore'] = (df_eng['lot_occupancy_ratio'] - global_mean) / (global_std + eps)
df_eng['slot_vs_lot_deviation'] = df_eng['label'] - df_eng['lot_occupancy_ratio']

print('Feature engineering done.')
print(f'Total numeric features: {df_eng.select_dtypes(include=np.number).shape[1] - 1}')
# CELL

drop_cols = ['label', 'lot', 'weather', 'filename', 'slot_idx', 'slot_row',
             'slot_col', 'dist_from_entrance']
drop_cols = [c for c in drop_cols if c in df_eng.columns]

X = df_eng.drop(columns=drop_cols, errors='ignore').select_dtypes(include=np.number)
y = df_eng['label'].astype(int)

print(f'X: {X.shape} | y: {y.shape} | positive rate: {y.mean():.2%}')


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.176, random_state=42, stratify=y_train_full)

print(f'\nTrain: {X_train.shape[0]:,} ({len(X_train)/len(X)*100:.0f}%)')
print(f'Val  : {X_val.shape[0]:,}  ({len(X_val)/len(X)*100:.0f}%)')
print(f'Test : {X_test.shape[0]:,}  ({len(X_test)/len(X)*100:.0f}%)')

scaler2     = StandardScaler()
X_train_sc  = scaler2.fit_transform(X_train)
X_val_sc    = scaler2.transform(X_val)
X_test_sc   = scaler2.transform(X_test)

print('\nSplits ready. Scaler fitted on training set only.')
# CELL
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
import time

MODEL_ZOO = {
    'Logistic Regression': LogisticRegression(
        C=1.0, max_iter=2000, solver='lbfgs', class_weight='balanced', random_state=42),
    'SVM (RBF)': SVC(
        kernel='rbf', C=5.0, gamma='scale', probability=True,
        class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, max_depth=20, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0),
    'LightGBM': LGBMClassifier(
        n_estimators=300, learning_rate=0.08, max_depth=8, num_leaves=63,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1),
}


SCALE_REQUIRED = {'Logistic Regression', 'SVM (RBF)'}

cv_strat   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
CV_SCORING = ['f1', 'roc_auc', 'accuracy', 'precision', 'recall']

print(f'Training {len(MODEL_ZOO)} models...')
# CELL

from sklearn.model_selection import cross_validate

print(f'{"Model":<25} {"CV F1":>8} {"±":>6} {"CV AUC":>8} {"±":>6}')
print('─' * 60)

cv_results_raw = {}

for name, model in MODEL_ZOO.items():
    X_cv = X_train_sc if name in SCALE_REQUIRED else X_train.values
    t0   = time.time()
    cv_out = cross_validate(model, X_cv, y_train.values, cv=cv_strat,
                            scoring={m: m for m in CV_SCORING},
                            return_train_score=False, n_jobs=1)
    elapsed = time.time() - t0
    cv_results_raw[name] = {**cv_out, 'elapsed': elapsed}

    f1_m  = cv_out['test_f1'].mean()
    f1_s  = cv_out['test_f1'].std()
    auc_m = cv_out['test_roc_auc'].mean()
    auc_s = cv_out['test_roc_auc'].std()
    print(f'  {name:<23} {f1_m:>8.4f} {f1_s:>6.4f} {auc_m:>8.4f} {auc_s:>6.4f}  [{elapsed:.1f}s]')
# CELL

trained_models = {}
val_results    = []

print(f'Full fit on {X_train.shape[0]:,} samples, eval on {X_val.shape[0]:,}...')

for name, model in MODEL_ZOO.items():
    X_tr = X_train_sc if name in SCALE_REQUIRED else X_train.values
    X_vl = X_val_sc   if name in SCALE_REQUIRED else X_val.values

    t0 = time.time()
    model.fit(X_tr, y_train.values)
    trained_models[name] = model

    y_pred  = model.predict(X_vl)
    y_proba = model.predict_proba(X_vl)[:, 1] if hasattr(model, 'predict_proba') else None

    val_results.append({
        'Model'        : name,
        'Val F1'       : f1_score(y_val, y_pred),
        'Val AUC-ROC'  : roc_auc_score(y_val, y_proba) if y_proba is not None else np.nan,
        'Val Accuracy' : accuracy_score(y_val, y_pred),
        'Val MCC'      : matthews_corrcoef(y_val, y_pred),
        'Val Precision': __import__('sklearn.metrics', fromlist=['precision_score']).precision_score(y_val, y_pred, zero_division=0),
        'Val Recall'   : __import__('sklearn.metrics', fromlist=['recall_score']).recall_score(y_val, y_pred, zero_division=0),
        'CV F1 Mean'   : cv_results_raw[name]['test_f1'].mean(),
        'CV F1 Std'    : cv_results_raw[name]['test_f1'].std(),
        'Fit Time (s)' : round(time.time()-t0, 2),
    })
    row = val_results[-1]
    print(f'  {name:<25}  F1={row["Val F1"]:.4f}  AUC={row["Val AUC-ROC"]:.4f}  [{row["Fit Time (s)"]:.1f}s]')

val_df = pd.DataFrame(val_results).sort_values('Val F1', ascending=False).reset_index(drop=True)
print(f'\nBest model: {val_df.iloc[0]["Model"]}  F1={val_df.iloc[0]["Val F1"]:.4f}')
# CELL

metrics_plot = ['Val F1', 'Val AUC-ROC', 'Val MCC']
fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Comparison — Validation Set', fontsize=13, fontweight='bold')

order      = val_df['Model'].tolist()
bar_colors = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(order)))

for ax, metric in zip(axes, metrics_plot):
    vals = val_df.set_index('Model').loc[order, metric].values
    bars = ax.barh(order[::-1], vals[::-1], color=bar_colors, height=0.6, edgecolor='white')
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() - bar.get_width()*0.04, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', ha='right', va='center', fontsize=8.5,
                fontweight='bold', color='white')
    ax.set_title(metric.replace('Val ', ''), fontweight='bold')
    ax.set_xlabel('Score')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL

n_models = len(MODEL_ZOO)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Confusion Matrices — Validation Set', fontsize=13, fontweight='bold')
axes = axes.flatten()

for i, (name, model) in enumerate(MODEL_ZOO.items()):
    X_vl   = X_val_sc if name in SCALE_REQUIRED else X_val.values
    y_pred = model.predict(X_vl)
    cm     = confusion_matrix(y_val, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    axes[i].imshow(cm, interpolation='nearest', cmap='Blues')
    axes[i].set_title(f'{name}\nF1={f1_score(y_val,y_pred):.4f}', fontsize=9, fontweight='bold')
    thresh = cm.max() / 2
    for r in range(2):
        for c in range(2):
            axes[i].text(c, r, f'{cm[r,c]:,}\n({cm_pct[r,c]:.1f}%)',
                         ha='center', va='center', fontsize=10, fontweight='bold',
                         color='white' if cm[r,c] > thresh else 'black')
    axes[i].set_xticks([0,1]); axes[i].set_yticks([0,1])
    axes[i].set_xticklabels(['Empty','Occupied']); axes[i].set_yticklabels(['Empty','Occupied'])
    axes[i].set_xlabel('Predicted'); axes[i].set_ylabel('Actual')

for j in range(n_models, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL

from sklearn.metrics import roc_curve

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title('ROC Curves — Validation Set', fontsize=13, fontweight='bold')
roc_colors = plt.cm.tab10(np.linspace(0, 0.8, len(MODEL_ZOO)))

for i, (name, model) in enumerate(MODEL_ZOO.items()):
    if not hasattr(model, 'predict_proba'): continue
    X_vl  = X_val_sc if name in SCALE_REQUIRED else X_val.values
    proba = model.predict_proba(X_vl)[:, 1]
    auc   = roc_auc_score(y_val, proba)
    fpr, tpr, _ = roc_curve(y_val, proba)
    ax.plot(fpr, tpr, lw=2, color=roc_colors[i], label=f'{name} (AUC={auc:.4f})')

ax.plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.legend(fontsize=8.5, loc='lower right')
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL
from scipy.stats import uniform, randint, loguniform

top2_names = val_df['Model'].head(2).tolist()
print(f'Tuning: {top2_names}')

CV_TUNE  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_ITER_R = 50

STAGE1_DISTRIBUTIONS = {
    'LightGBM': {
        'n_estimators'     : randint(100, 600),
        'learning_rate'    : loguniform(1e-3, 0.3),
        'max_depth'        : randint(3, 12),
        'num_leaves'       : randint(20, 150),
        'min_child_samples': randint(5, 100),
        'subsample'        : uniform(0.5, 0.5),
        'colsample_bytree' : uniform(0.5, 0.5),
        'reg_alpha'        : loguniform(1e-4, 10),
        'reg_lambda'       : loguniform(1e-4, 10),
    },
    'XGBoost': {
        'n_estimators'    : randint(100, 600),
        'learning_rate'   : loguniform(1e-3, 0.3),
        'max_depth'       : randint(3, 10),
        'subsample'       : uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 10),
        'reg_alpha'       : loguniform(1e-4, 10),
        'reg_lambda'      : loguniform(1e-4, 10),
        'gamma'           : uniform(0, 0.5),
    },
}

stage1_results, stage1_best_params = {}, {}

for name in top2_names:
    if name not in STAGE1_DISTRIBUTIONS: continue
    print(f'\n--- Stage-1 RandomizedSearch: {name} ---')
    print(f'  {N_ITER_R} configs × 5 folds = {N_ITER_R*5} fits...')

    base = MODEL_ZOO[name].__class__(**{
        k: v for k, v in MODEL_ZOO[name].get_params().items()
        if k in ['random_state', 'n_jobs', 'verbosity', 'verbose', 'eval_metric', 'class_weight']
    })
    X_tr = X_train_sc if name in SCALE_REQUIRED else X_train.values

    rscv = RandomizedSearchCV(base, STAGE1_DISTRIBUTIONS[name], n_iter=N_ITER_R,
                              scoring='f1', cv=CV_TUNE, refit=True, n_jobs=-1,
                              random_state=42, verbose=0)
    t0 = time.time()
    rscv.fit(X_tr, y_train.values)
    elapsed = time.time() - t0

    stage1_results[name]     = rscv
    stage1_best_params[name] = rscv.best_params_

    X_vl   = X_val_sc if name in SCALE_REQUIRED else X_val.values
    y_pred  = rscv.best_estimator_.predict(X_vl)
    val_f1  = f1_score(y_val, y_pred)
    base_f1 = val_df[val_df['Model'] == name]['Val F1'].values[0]

    print(f'  CV F1: {rscv.best_score_:.4f} | Val F1: {val_f1:.4f} ({val_f1 - base_f1:+.4f} vs baseline) | {elapsed:.1f}s')
# CELL

def make_stage2_grid(name, best_params):
    grid = {}
    if name in ('LightGBM', 'XGBoost'):
        lr, ne, md = best_params.get('learning_rate', 0.1), best_params.get('n_estimators', 300), best_params.get('max_depth', 6)
        grid['learning_rate'] = sorted(set([round(max(0.001, lr*0.6), 4), round(lr, 4), round(lr*1.5, 4)]))
        grid['n_estimators']  = sorted(set([max(50, int(ne*0.7)), int(ne), int(ne*1.3)]))
        grid['max_depth']     = sorted(set([max(2, md-1), md, md+1]))
        for k in best_params:
            if k not in grid: grid[k] = [best_params[k]]
    return grid

stage2_results = {}
for name in top2_names:
    if name not in stage1_best_params: continue
    print(f'\n--- Stage-2 GridSearch: {name} ---')
    s2_grid = make_stage2_grid(name, stage1_best_params[name])
    n_combos = 1
    for v in s2_grid.values(): n_combos *= len(v)
    print(f'  {n_combos} combinations × 5 folds = {n_combos*5} fits...')

    s1_fixed = {k: v for k, v in stage1_best_params[name].items() if k not in s2_grid}
    base = MODEL_ZOO[name].__class__(**{
        **{k: v for k, v in MODEL_ZOO[name].get_params().items()
           if k in ['random_state', 'n_jobs', 'verbosity', 'verbose', 'eval_metric', 'class_weight']},
        **s1_fixed
    })
    X_tr = X_train_sc if name in SCALE_REQUIRED else X_train.values

    gscv = GridSearchCV(base, s2_grid, scoring='f1', cv=CV_TUNE, refit=True, n_jobs=-1, verbose=0)
    t0 = time.time()
    gscv.fit(X_tr, y_train.values)
    stage2_results[name] = gscv

    X_vl   = X_val_sc if name in SCALE_REQUIRED else X_val.values
    y_pred  = gscv.best_estimator_.predict(X_vl)
    val_f1  = f1_score(y_val, y_pred)
    base_f1 = val_df[val_df['Model'] == name]['Val F1'].values[0]
    print(f'  CV F1: {gscv.best_score_:.4f} | Val F1: {val_f1:.4f} ({val_f1 - base_f1:+.4f} total gain) | {time.time()-t0:.1f}s')

tuned_models = {name: stage2_results[name].best_estimator_ for name in stage2_results}
print(f'\nTuned models ready: {list(tuned_models.keys())}')
# CELL
from sklearn.metrics import classification_report, precision_score, recall_score

# pick champion
champion_name  = list(tuned_models.keys())[0]
champion_model = tuned_models[champion_name]

X_test_in    = X_test_sc if champion_name in SCALE_REQUIRED else X_test.values
y_pred_test  = champion_model.predict(X_test_in)
y_proba_test = champion_model.predict_proba(X_test_in)[:,1]

print('=' * 60)
print(f'CHAMPION MODEL: {champion_name}')
print('=' * 60)
print('\nClassification Report (Test Set):')
print(classification_report(y_test, y_pred_test, target_names=['Empty','Occupied'], digits=4))
print(f'  AUC-ROC : {roc_auc_score(y_test, y_proba_test):.4f}')
print(f'  F1 Score: {f1_score(y_test, y_pred_test):.4f}')
print(f'  Accuracy: {accuracy_score(y_test, y_pred_test):.4f}')
# CELL

cm      = confusion_matrix(y_test, y_pred_test)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
labels  = ['Empty (0)', 'Occupied (1)']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f'Confusion Matrix — {champion_name} (Test Set)', fontsize=13, fontweight='bold')

for ax, data, title in [(axes[0], cm, 'Raw Counts'), (axes[1], cm_norm, 'Row-Normalised (%)')]:
    ax.imshow(data, interpolation='nearest', cmap='Blues')
    thresh = data.max() / 2
    for i in range(2):
        for j in range(2):
            val_str = f'{data[i,j]:,}' if data is cm else f'{data[i,j]*100:.1f}%'
            ax.text(j, i, val_str, ha='center', va='center', fontsize=16, fontweight='bold',
                    color='white' if data[i,j] > thresh else 'black')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels, rotation=45)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title, fontweight='bold')

plt.tight_layout()
plt.savefig('final_confusion_matrix.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f'TP={tp:,}  FP={fp:,}  FN={fn:,}  TN={tn:,}')
# CELL

from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f'Final Curves — {champion_name}', fontsize=13, fontweight='bold')

prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba_test)
ap_c = average_precision_score(y_test, y_proba_test)
fpr, tpr, thr = roc_curve(y_test, y_proba_test)
auc_val = roc_auc_score(y_test, y_proba_test)

axes[0].plot(fpr, tpr, lw=2.5, color='#8E44AD', label=f'AUC={auc_val:.4f}')
axes[0].fill_between(fpr, tpr, alpha=0.1, color='#8E44AD')
axes[0].plot([0,1],[0,1],'k--',lw=1,alpha=0.5)
axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curve', fontweight='bold'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(rec_c, prec_c, lw=2.5, color='#E74C3C', label=f'AP={ap_c:.4f}')
axes[1].axhline(y_test.mean(), ls='--', lw=1.5, color='grey', label=f'Baseline ({y_test.mean():.2f})')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curve', fontweight='bold'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('final_roc_pr.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print(f'AUC-ROC: {auc_val:.4f} | Average Precision: {ap_c:.4f}')
# CELL

thresholds = np.arange(0.1, 0.9, 0.01)
f1s, accs  = [], []

for t in thresholds:
    yp = (y_proba_test >= t).astype(int)
    f1s.append(f1_score(y_test, yp))
    accs.append(accuracy_score(y_test, yp))

best_t = thresholds[np.argmax(f1s)]

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(thresholds, f1s,  lw=2, label='F1',      color='#8E44AD')
ax.plot(thresholds, accs, lw=2, label='Accuracy', color='#3498DB')
ax.axvline(best_t, color='orange', ls=':', lw=2.5, label=f'Best F1 threshold={best_t:.2f}')
ax.axvline(0.5,    color='gray',   ls=':', lw=1.5, alpha=0.7, label='Default (0.5)')
ax.set_xlabel('Decision Threshold')
ax.set_ylabel('Score')
ax.set_title('Threshold Sensitivity', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
print(f'Optimal threshold: {best_t:.2f}  →  F1 = {max(f1s):.4f}')
# CELL

top3 = val_df['Model'].head(3).tolist()
print(f'Building ensemble from: {top3}')

ensemble_estimators = [(name, trained_models[name]) for name in top3]
voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)
voting_clf.fit(X_train.values, y_train.values)

y_ens_pred  = voting_clf.predict(X_test.values)
y_ens_proba = voting_clf.predict_proba(X_test.values)[:,1]

print('\nSoft Voting Ensemble — Test Set:')
print(f'  Accuracy: {accuracy_score(y_test, y_ens_pred):.4f}')
print(f'  F1 Score: {f1_score(y_test, y_ens_pred):.4f}')
print(f'  AUC-ROC : {roc_auc_score(y_test, y_ens_proba):.4f}')
# CELL
feature_names = list(X_train.columns)
X_test_in     = X_test_sc if champion_name in SCALE_REQUIRED else X_test.values


if hasattr(champion_model, 'feature_importances_'):
    fi = pd.Series(champion_model.feature_importances_, index=feature_names)
    fi_method = 'Built-in (Gini/Gain)'
elif hasattr(champion_model, 'coef_'):
    fi = pd.Series(np.abs(champion_model.coef_[0]), index=feature_names)
    fi_method = '|Coefficient|'
else:
    fi = None


perm_result = permutation_importance(champion_model, X_test_in, y_test,
                                     n_repeats=15, random_state=42, scoring='f1')
perm_fi  = pd.Series(perm_result.importances_mean, index=feature_names)
perm_std = pd.Series(perm_result.importances_std,  index=feature_names)

TOP_N = 20
fig, axes = plt.subplots(1, 2 if fi is not None else 1,
                          figsize=(14 if fi is not None else 8, 7))
fig.suptitle(f'Feature Importance — {champion_name}', fontsize=13, fontweight='bold')
viridis = plt.cm.viridis(np.linspace(0.15, 0.9, TOP_N))[::-1]

if fi is not None:
    top_bi = fi.nlargest(TOP_N)
    axes[0].barh(range(TOP_N), top_bi.values[::-1], color=viridis, edgecolor='white')
    axes[0].set_yticks(range(TOP_N))
    axes[0].set_yticklabels([f.replace('_',' ') for f in top_bi.index[::-1]], fontsize=8.5)
    axes[0].set_title(f'{fi_method}', fontweight='bold')
    ax_pi = axes[1]
else:
    ax_pi = axes

top_perm = perm_fi.nlargest(TOP_N)
ax_pi.barh(range(TOP_N), top_perm.values[::-1],
           xerr=perm_std[top_perm.index].values[::-1],
           color=viridis, edgecolor='white', capsize=3)
ax_pi.set_yticks(range(TOP_N))
ax_pi.set_yticklabels([f.replace('_',' ') for f in top_perm.index[::-1]], fontsize=8.5)
ax_pi.set_title('Permutation Importance (Test Set)', fontweight='bold')
ax_pi.axvline(0, color='black', lw=0.8)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()

print('\nTop 10 features (permutation):')
print(perm_fi.nlargest(10).round(5).to_string())
# CELL

feat_groups = {
    'Color Statistics':     [c for c in perm_fi.index if 'color' in c or c in ['brightness','saturation_mean']],
    'HOG (Shape/Edge)':     [c for c in perm_fi.index if 'hog' in c],
    'LBP (Texture)':        [c for c in perm_fi.index if 'lbp' in c],
    'GLCM (Co-occurrence)': [c for c in perm_fi.index if 'glcm' in c],
    'Edge & Sharpness':     [c for c in perm_fi.index if c in ['edge_density','laplacian_var','contrast']],
    'Engineered Features':  [c for c in perm_fi.index if any(x in c for x in ['ratio','uniformity','score','energy','entropy','bright_contrast'])],
    'Metadata':             [c for c in perm_fi.index if any(x in c for x in ['lot','weather','wx_'])],
    'Domain Features':      [c for c in perm_fi.index if any(x in c for x in ['dist_from','occupancy','neighbour','dark_slot','brightness_z','relative_b'])],
}

group_imp = {g: perm_fi[cols].sum() for g, cols in feat_groups.items() if cols}
group_s   = pd.Series(group_imp).sort_values()

colors_g = plt.cm.Set2(np.linspace(0, 1, len(group_s)))
fig, ax  = plt.subplots(figsize=(10, 5))
ax.barh(group_s.index, group_s.values, color=colors_g, edgecolor='white', height=0.6)
ax.set_xlabel('Total Permutation Importance (F1 drop)', fontsize=11)
ax.set_title('Feature Importance by Group', fontsize=13, fontweight='bold')
for i, (idx, val) in enumerate(group_s.items()):
    ax.text(val + 0.0002, i, f'{val:.4f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig('feature_importance_groups.png', dpi=FIG_DPI, bbox_inches='tight')
plt.show()
# CELL
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train_sc, y_train)
y_pred_gnb  = gnb.predict(X_val_sc)
y_proba_gnb = gnb.predict_proba(X_val_sc)[:, 1]

gnb_f1  = f1_score(y_val, y_pred_gnb)
gnb_auc = roc_auc_score(y_val, y_proba_gnb)
print(f'Naive Bayes — Val F1: {gnb_f1:.4f} | AUC: {gnb_auc:.4f}')
print(classification_report(y_val, y_pred_gnb, target_names=['Empty', 'Occupied']))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle('Experiment 1: Gaussian Naive Bayes', fontsize=12, fontweight='bold')

cm_gnb = confusion_matrix(y_val, y_pred_gnb)
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Empty','Occupied'], yticklabels=['Empty','Occupied'])
axes[0].set_title('Confusion Matrix'); axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

axes[1].hist(y_proba_gnb[y_val == 0], bins=40, alpha=0.6, color='#2ECC71', label='True Empty', density=True)
axes[1].hist(y_proba_gnb[y_val == 1], bins=40, alpha=0.6, color='#E74C3C', label='True Occupied', density=True)
axes[1].axvline(0.5, color='black', ls='--', lw=1.5)
axes[1].set_xlabel('P(Occupied)'); axes[1].set_ylabel('Density')
axes[1].set_title('Predicted Probability Distribution'); axes[1].legend()
plt.tight_layout()
plt.savefig('exp1_naive_bayes.png', dpi=120); plt.show()
# CELL
from sklearn.neighbors import KNeighborsClassifier

k_values    = [5, 15, 51]
knn_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean', weights='distance', n_jobs=-1)
    knn.fit(X_train_sc, y_train)
    t0 = time.time()
    y_pred_knn  = knn.predict(X_val_sc)
    y_proba_knn = knn.predict_proba(X_val_sc)[:, 1]
    pred_ms     = (time.time() - t0) * 1000
    knn_results.append({'k': k, 'f1': f1_score(y_val, y_pred_knn),
                         'auc': roc_auc_score(y_val, y_proba_knn),
                         'pred_time_ms': pred_ms, 'model': knn})
    print(f'  k={k:<3}  F1={knn_results[-1]["f1"]:.4f}  AUC={knn_results[-1]["auc"]:.4f}  [{pred_ms:.1f}ms]')

best_knn = max(knn_results, key=lambda r: r['f1'])
print(f'\nBest KNN: k={best_knn["k"]}  F1={best_knn["f1"]:.4f}')
# CELL
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

tree_configs = [
    {'label': 'Unpruned',    'max_depth': None, 'min_samples_leaf': 1},
    {'label': 'Pruned (d=6)','max_depth': 6,    'min_samples_leaf': 10},
    {'label': 'Pruned (d=4)','max_depth': 4,    'min_samples_leaf': 20},
]

tree_results = []
for cfg in tree_configs:
    dt = DecisionTreeClassifier(max_depth=cfg['max_depth'],
                                min_samples_leaf=cfg['min_samples_leaf'],
                                class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)  
    y_pred_dt  = dt.predict(X_val)
    y_proba_dt = dt.predict_proba(X_val)[:, 1]
    train_f1   = f1_score(y_train, dt.predict(X_train))
    val_f1     = f1_score(y_val, y_pred_dt)
    tree_results.append({'label': cfg['label'], 'train_f1': train_f1, 'val_f1': val_f1,
                          'val_auc': roc_auc_score(y_val, y_proba_dt),
                          'depth': dt.get_depth(), 'n_leaves': dt.get_n_leaves(), 'model': dt})
    print(f'  {cfg["label"]:<15}  TrainF1={train_f1:.4f}  ValF1={val_f1:.4f}  Depth={dt.get_depth()}  Leaves={dt.get_n_leaves()}')


feature_names_short = [f[:18] for f in list(X_train.columns)]
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Experiment 3: Single Decision Tree', fontsize=12, fontweight='bold')

x_pos = np.arange(len(tree_results)); w = 0.35
axes[0].bar(x_pos - w/2, [r['train_f1'] for r in tree_results], width=w,
            label='Train F1', color='#3498DB', edgecolor='white', alpha=0.85)
axes[0].bar(x_pos + w/2, [r['val_f1'] for r in tree_results], width=w,
            label='Val F1',   color='#E74C3C', edgecolor='white', alpha=0.85)
axes[0].axhline(0.96, color='black', ls='--', lw=1.2, label='XGBoost baseline')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels([r['label'] for r in tree_results], fontsize=9)
axes[0].set_ylabel('F1 Score'); axes[0].set_title('Train vs Val F1', fontweight='bold')
axes[0].legend(); axes[0].set_ylim(0.75, 1.02)

plot_tree(tree_results[2]['model'], ax=axes[1],
          feature_names=feature_names_short, class_names=['Empty', 'Occupied'],
          filled=True, rounded=True, max_depth=3, fontsize=7, impurity=False, proportion=True)
axes[1].set_title('Depth-4 Tree (top 3 levels)', fontweight='bold')
plt.tight_layout(); plt.savefig('exp3_decision_tree.png', dpi=120); plt.show()

print('\nTop of depth-4 tree rules:')
print(export_text(tree_results[2]['model'], feature_names=feature_names_short, max_depth=3))
# CELL

exp_models = [
    ('Naive Bayes',         gnb_f1,              gnb_auc),
    ('KNN (k=15)',          best_knn['f1'],       best_knn['auc']),
    ('Decision Tree (d=6)', tree_results[1]['val_f1'], tree_results[1]['val_auc']),
    ('XGBoost (champion)',  0.96,                 0.99),
]

labels_exp = [m[0] for m in exp_models]
f1s_exp    = [m[1] for m in exp_models]
aucs_exp   = [m[2] for m in exp_models]
colors_exp = ['#95A5A6', '#95A5A6', '#95A5A6', '#E74C3C']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Experiment Models vs Champion — Validation Set', fontsize=13, fontweight='bold')

for ax, vals, ylabel in [(axes[0], f1s_exp, 'F1 Score'), (axes[1], aucs_exp, 'AUC-ROC')]:
    bars = ax.bar(labels_exp, vals, color=colors_exp, edgecolor='white', width=0.55)
    ax.set_ylabel(ylabel); ax.set_ylim(0.70, 1.02); ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=12)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('exp_summary.png', dpi=120); plt.show()
print('XGBoost remains champion.')
# CELL
model_artifact = {
    'model':         champion_model,
    'scaler':        scaler2,
    'feature_names': list(X_train.columns),
    'threshold':     best_t,
    'model_name':    champion_name
}
joblib.dump(model_artifact, 'smart_drone_parking_model.pkl')
print('Champion model saved to smart_drone_parking_model.pkl')
# CELL
final_f1  = f1_score(y_test, y_pred_test)
final_acc = accuracy_score(y_test, y_pred_test)
final_auc = roc_auc_score(y_test, y_proba_test)

y_pred_opt = (y_proba_test >= best_t).astype(int)
opt_f1     = f1_score(y_test, y_pred_opt)
opt_acc    = accuracy_score(y_test, y_pred_opt)

print('=' * 60)
print('  SMART DRONE PARKING AI — FINAL RESULTS')
print('=' * 60)
print(f'  Champion   : {champion_name}')
print(f'  Test set   : {len(y_test):,} samples')
print()
print(f'  Default threshold (0.50):')
print(f'    Accuracy : {final_acc:.4f}')
print(f'    F1 Score : {final_f1:.4f}')
print(f'    AUC-ROC  : {final_auc:.4f}')
print()
print(f'  Optimal threshold ({best_t:.2f}):')
print(f'    Accuracy : {opt_acc:.4f}')
print(f'    F1 Score : {opt_f1:.4f}')
print()
print('  Target KPIs:')
print(f'    F1  ≥ 0.95: {"PASS" if final_f1  >= 0.95 else "FAIL"} ({final_f1:.4f})')
print(f'    Acc ≥ 95% : {"PASS" if final_acc >= 0.95 else "FAIL"} ({final_acc*100:.2f}%)')
print(f'    AUC ≥ 0.98: {"PASS" if final_auc >= 0.98 else "FAIL"} ({final_auc:.4f})')
print('=' * 60)