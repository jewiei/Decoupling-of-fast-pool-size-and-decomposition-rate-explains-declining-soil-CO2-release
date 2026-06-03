"""
Title: Re-analysis of Mu & Mu (2024/2025) Thermokarst Dataset
Source: PANGAEA.970424 (https://doi.org/10.1594/PANGAEA.970424)
Purpose: To disentangle pool-size vs. rate-control effects on CO2 emissions.
Author: Tingan Dai et al.
Date: 2026-05-31
Note: This script performs secondary analysis on publicly archived data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from itertools import combinations
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
})
sns.set_style("ticks")
sns.set_palette("deep")

# ======================== 1. 数据读取与预处理 ========================
file_path = "/Users/daitingan/Desktop/小论文/五/11111.xlsx"
output_dir = "/Users/daitingan/Desktop/小论文/五/"

df1 = pd.read_excel(file_path, sheet_name='Sheet1')
df2 = pd.read_excel(file_path, sheet_name='Sheet2')
df3 = pd.read_excel(file_path, sheet_name='Sheet3')

df1.columns = [c.strip() for c in df1.columns]
df2.columns = [c.strip() for c in df2.columns]
df3.columns = [c.strip() for c in df3.columns]

df2['Years'] = df2['Years'].replace('10 years', '12 years')

year_order = ['Control', '1 year', '12 years', '23 years']
df1['Years'] = pd.Categorical(df1['Years'], categories=year_order, ordered=True)
df2['Years'] = pd.Categorical(df2['Years'], categories=year_order, ordered=True)
df3['Years'] = pd.Categorical(df3['Years'], categories=year_order, ordered=True)

df1.rename(columns={'Depth comm': 'Depth'}, inplace=True)
df2.rename(columns={'Depth comm': 'Depth'}, inplace=True)
df3.rename(columns={'Depth comm': 'Depth'}, inplace=True)

df1['SampleID'] = df1['Sample label']
df1 = df1.merge(df3, on=['Sample label', 'Years', 'Depth'], how='left', suffixes=('', '_prop'))
df1 = df1.loc[:, ~df1.columns.duplicated()]

numeric_cols = ['SOC [g/kg]', 'TN [g/kg]', 'C/N', 'DOC [mg/kg]',
                'O-alkyl C [%]', 'Alkyl C [%]', 'Aromatics [%]', 'Carboxyl [%]',
                'Alkyl C/O-alkyl C', 'PLFA [µg/g] (Actinobacteria)', 'PLFA [µg/g] (fungi)',
                'PLFA [µg/g] (Bacteria)', 'PLFA [µg/g] (Total)', 'fungalPLFA/bacterialPLFA',
                'MUF-Phos [nmol/g/h]', 'NAG [nmol/g/h]', 'LAP [nmol/g/h]', 'BG [nmol/g/h]']
for col in numeric_cols:
    if col in df1.columns:
        df1[col] = pd.to_numeric(df1[col], errors='coerce')

df1['Group'] = df1['Years'].astype(str) + '_' + df1['Depth']

# ======================== 2. 模型拟合函数（修正版：AICc + 边界检查） ========================
def single_pool(t, C0, k):
    return C0 * (1 - np.exp(-k * t))

def double_pool(t, C0_fast, k_fast, C0_slow, k_slow):
    return (C0_fast * (1 - np.exp(-k_fast * t)) +
            C0_slow * (1 - np.exp(-k_slow * t)))

def calculate_aicc(n, ss_res, k):
    """计算AICc（小样本校正AIC）"""
    if n <= k + 1:
        return np.inf
    aic = n * np.log(ss_res / n) + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return aicc

def fit_models(sample_data):
    t = sample_data['Day'].values
    y = sample_data['CO2-C/SOC aerob [mg/g]'].values
    n = len(t)
    if n < 4:
        return None
    
    C0_init = y.max() * 1.1
    k_init = 0.05
    
    # 单库模型拟合
    try:
        popt_single, _ = curve_fit(single_pool, t, y, p0=[C0_init, k_init],
                                   bounds=(0, [np.inf, 1]), maxfev=10000)
        y_pred_single = single_pool(t, *popt_single)
        ss_res = np.sum((y - y_pred_single)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2_single = 1 - ss_res / ss_tot
        aic_single = calculate_aicc(n, ss_res, k=3)
    except:
        popt_single = [np.nan, np.nan]
        y_pred_single = None
        r2_single = -np.inf
        aic_single = np.inf

    # 双库模型拟合
    try:
        p0_double = [C0_init*0.3, 0.2, C0_init*0.7, 0.01]
        bounds = ([0, 0, 0, 0], [np.inf, 1.0, np.inf, 0.5])
        popt_double, _ = curve_fit(double_pool, t, y, p0=p0_double,
                                   bounds=bounds, maxfev=10000)
        y_pred_double = double_pool(t, *popt_double)
        ss_res = np.sum((y - y_pred_double)**2)
        r2_double = 1 - ss_res / ss_tot
        aic_double = calculate_aicc(n, ss_res, k=5)
        
        hit_boundary = False
        if abs(popt_double[1] - 1.0) < 1e-6 or abs(popt_double[3] - 0.5) < 1e-6:
            hit_boundary = True
    except:
        popt_double = [np.nan, np.nan, np.nan, np.nan]
        y_pred_double = None
        r2_double = -np.inf
        aic_double = np.inf
        hit_boundary = False

    delta_aic = aic_single - aic_double
    
    if delta_aic <= 2:
        best_model = 'single'
        best_params = {
            'C0_total': popt_single[0], 'k_total': popt_single[1],
            'C0_fast': np.nan, 'k_fast': np.nan,
            'C0_slow': np.nan, 'k_slow': np.nan,
            'R2': r2_single, 'AICc': aic_single, 'Delta_AICc': delta_aic,
            'Hit_boundary': False
        }
    else:
        best_model = 'double'
        best_params = {
            'C0_total': popt_double[0] + popt_double[2],
            'k_total': np.nan,
            'C0_fast': popt_double[0], 'k_fast': popt_double[1],
            'C0_slow': popt_double[2], 'k_slow': popt_double[3],
            'R2': r2_double, 'AICc': aic_double, 'Delta_AICc': delta_aic,
            'Hit_boundary': hit_boundary
        }
    
    return {
        'model': best_model, 'params': best_params,
        't': t, 'y': y, 'y_pred_single': y_pred_single, 'y_pred_double': y_pred_double,
        'r2_single': r2_single, 'r2_double': r2_double,
        'aic_single': aic_single, 'aic_double': aic_double
    }

# ======================== 3. 批量拟合 ========================
sample_ids = df1['SampleID'].unique()
fit_results = []
fit_details = {}
for sid in sample_ids:
    sub = df1[df1['SampleID'] == sid].copy()
    years = sub['Years'].iloc[0]; depth = sub['Depth'].iloc[0]
    sub_data = sub[['Day', 'CO2-C/SOC aerob [mg/g]']].dropna()
    if len(sub_data) < 4: continue
    result = fit_models(sub_data)
    if result is None: continue
    rec = {'SampleID': sid, 'Years': years, 'Depth': depth,
           'Group': f"{years}_{depth}", 'Best_model': result['model']}
    rec.update(result['params'])
    fit_results.append(rec)
    result['Years'] = years
    result['Depth'] = depth
    fit_details[sid] = result

df_fit = pd.DataFrame(fit_results)
df_fit.to_csv(output_dir + "fitting_parameters.csv", index=False)

# ================== 生成模型选择文件 model_selection.csv ==================
model_selection = []
for sid, det in fit_details.items():
    aic_s = det['aic_single']
    aic_d = det['aic_double']
    delta = aic_s - aic_d
    model_selection.append({
        'SampleID': sid,
        'Years': det['Years'],
        'Depth': det['Depth'],
        'AICc_single': aic_s,
        'AICc_double': aic_d,
        'Delta_AICc': delta,
        'Selected_model': det['model']
    })
df_model_sel = pd.DataFrame(model_selection)
df_model_sel.to_csv(output_dir + "model_selection.csv", index=False)
print("Model selection table saved: model_selection.csv")

# ======================== 4. 方差分析与效应量 ========================
def anova_tukey(data, dep_var, factors=['Years', 'Depth']):
    sub_data = data.dropna(subset=[dep_var])
    print(f"   [ANOVA for {dep_var}: n={len(sub_data)}]")
    formula = f'Q("{dep_var}") ~ C(Years) + C(Depth) + C(Years):C(Depth)'
    model = ols(formula, data=sub_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(f"\n========= ANOVA for {dep_var} =========")
    print(anova_table)
    anova_table.to_csv(output_dir + f"ANOVA_{dep_var}.csv")
    if anova_table.loc['C(Years):C(Depth)', 'PR(>F)'] < 0.05:
        for depth in sub_data['Depth'].unique():
            sub = sub_data[sub_data['Depth'] == depth]
            tukey = pairwise_tukeyhsd(sub[dep_var], sub['Years'])
            print(f"\nDepth={depth}:\n", tukey)
            with open(output_dir + f"tukey_{dep_var}_{depth}.txt", 'w') as f:
                f.write(str(tukey))
    else:
        for factor in ['Years', 'Depth']:
            if anova_table.loc[f'C({factor})', 'PR(>F)'] < 0.05:
                tukey = pairwise_tukeyhsd(sub_data[dep_var], sub_data[factor])
                print(f"\n{factor}:\n", tukey)
                with open(output_dir + f"tukey_{dep_var}_{factor}.txt", 'w') as f:
                    f.write(str(tukey))
    return anova_table

anova_results = {}
for var in ['C0_fast', 'k_fast', 'C0_total', 'C0_slow', 'k_slow']:
    if var in df_fit.columns:
        anova_results[var] = anova_tukey(df_fit, var)

anova_combined = pd.concat(anova_results, axis=0)
anova_combined.to_csv(output_dir + "Table2_ANOVA_summary.csv")

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

effect_sizes = []
for g1, g2 in [('23 years', '1 year'), ('23 years', '12 years'), ('23 years', 'Control')]:
    x = df_fit[df_fit['Years'] == g1]['k_fast'].dropna()
    y = df_fit[df_fit['Years'] == g2]['k_fast'].dropna()
    d = cohens_d(x, y)
    effect_sizes.append({'Comparison': f'{g1} vs {g2}', "Cohen's d": d})
    print(f"Cohen's d: {g1} vs {g2} = {d:.3f}")
pd.DataFrame(effect_sizes).to_csv(output_dir + "effect_sizes.csv", index=False)

# ======================== 5. 图形 ========================
est_colors = sns.color_palette("deep", 4)

# --- 5.1 累积矿化拟合图 (Fig. 2) ---
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
for idx, group_name in enumerate(['Control_Topsoil', 'Control_Subsoil',
                                   '1 year_Topsoil', '1 year_Subsoil',
                                   '12 years_Topsoil', '12 years_Subsoil',
                                   '23 years_Topsoil', '23 years_Subsoil']):
    ax = axes[idx]
    group_sids = df1[df1['Group'] == group_name]['SampleID'].unique()
    for i, sid in enumerate(group_sids):
        sub = df1[df1['SampleID'] == sid].sort_values('Day')
        t = sub['Day'].values
        y = sub['CO2-C/SOC aerob [mg/g]'].values
        color = est_colors[i % len(est_colors)]
        ax.scatter(t, y, color=color, s=25, edgecolors='white', linewidth=0.3, zorder=5)
        if sid in fit_details:
            det = fit_details[sid]
            t_fine = np.linspace(0, max(t), 100)
            if det['model'] == 'single' and det['y_pred_single'] is not None:
                ax.plot(t_fine, single_pool(t_fine, det['params']['C0_total'], det['params']['k_total']),
                        color=color, alpha=0.7, linewidth=1.5, zorder=4)
            elif det['model'] == 'double' and det['y_pred_double'] is not None:
                ax.plot(t_fine, double_pool(t_fine, *[det['params']['C0_fast'], det['params']['k_fast'],
                                                      det['params']['C0_slow'], det['params']['k_slow']]),
                        color=color, alpha=0.7, linewidth=1.5, zorder=4)
    years_part, depth_part = group_name.split('_')
    ax.set_title(f'{years_part} ({depth_part})', fontsize=14, fontweight='bold')
    ax.set_xlabel('Incubation day', fontsize=12)
    ax.set_ylabel('Cumulative CO$_{2}$-C\n(mg g$^{-1}$ SOC)', fontsize=12)
    ax.tick_params(labelsize=10)
fig.tight_layout(pad=2.0)
plt.savefig(output_dir + "Fig2_cumulative_fit.png", dpi=600)
plt.close()

# --- 5.2 双库柱状图 (Supplementary) ---
df_plot = df_fit.dropna(subset=['C0_fast', 'C0_slow']).copy()
df_plot['C0_resistant'] = (df_plot['C0_total'] - df_plot['C0_fast'] - df_plot['C0_slow']).clip(lower=0)
df_melt = df_plot.melt(id_vars=['Years', 'Depth'], value_vars=['C0_fast', 'C0_slow', 'C0_resistant'],
                       var_name='Pool', value_name='Size')
df_summary = df_melt.groupby(['Years', 'Depth', 'Pool'])['Size'].agg(['mean', 'std', 'count']).reset_index()
df_summary['se'] = df_summary['std'] / np.sqrt(df_summary['count'])

g = sns.catplot(data=df_summary, x='Years', y='mean', hue='Pool', col='Depth',
                kind='bar', palette='deep', height=4, aspect=1.2,
                ci=None, edgecolor='black', linewidth=0.5)
g.set_axis_labels('Recovery year', 'Potential mineralizable C\n(mg CO$_{2}$-C g$^{-1}$ SOC)')
g.set_titles('{col_name}')
plt.savefig(output_dir + "FigS_pool_bars.png", dpi=600)
plt.close()

# --- 5.3 酶化学计量图 (Supplementary) ---
df_enzyme = df1.dropna(subset=['BG [nmol/g/h]', 'NAG [nmol/g/h]', 'LAP [nmol/g/h]', 'MUF-Phos [nmol/g/h]']).copy()
df_enzyme['C_enz'] = df_enzyme['BG [nmol/g/h]']
df_enzyme['N_enz'] = df_enzyme['NAG [nmol/g/h]'] + df_enzyme['LAP [nmol/g/h]']
df_enzyme['P_enz'] = df_enzyme['MUF-Phos [nmol/g/h]']
df_enzyme['C_N_ratio'] = df_enzyme['C_enz'] / df_enzyme['N_enz']
df_enzyme['C_P_ratio'] = df_enzyme['C_enz'] / df_enzyme['P_enz']

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=df_enzyme, x='C_N_ratio', y='C_P_ratio', hue='Years', style='Depth',
                s=60, ax=ax, palette='deep', edgecolor='black', linewidth=0.3)
ax.axhline(1, color='gray', ls='--', lw=0.8)
ax.axvline(1, color='gray', ls='--', lw=0.8)
ax.set_xlabel('C:N enzyme ratio (BG/(NAG+LAP))', fontsize=12)
ax.set_ylabel('C:P enzyme ratio (BG/AP)', fontsize=12)
ax.set_title('Extracellular enzyme stoichiometry', fontsize=14, fontweight='bold')
ax.legend(loc='best', frameon=True, fancybox=True, fontsize=10)
plt.tight_layout()
plt.savefig(output_dir + "FigS_enzyme_stoichiometry.png", dpi=600)
plt.close()

# --- 5.4 相关性热图 (Fig. 4) ---
corr_vars_original = ['C0_total', 'C0_fast', 'k_fast', 'C0_slow', 'k_slow',
                      'SOC [g/kg]', 'TN [g/kg]', 'C/N', 'DOC [mg/kg]',
                      'O-alkyl C [%]', 'Alkyl C [%]', 'Alkyl C/O-alkyl C',
                      'PLFA [µg/g] (Total)', 'fungalPLFA/bacterialPLFA',
                      'BG [nmol/g/h]', 'NAG [nmol/g/h]']

var_labels = [
    r'$C_{0,total}$', r'$C_{0,fast}$', r'$k_{fast}$', r'$C_{0,slow}$', r'$k_{slow}$',
    'SOC', 'TN', 'C/N', 'DOC',
    'O-alkyl C', 'Alkyl C', 'Alkyl/O-alkyl',
    'Total PLFA', 'F/B ratio', 'BG', 'NAG'
]

df_corr = df_fit.merge(df3, left_on=['SampleID', 'Years', 'Depth'],
                       right_on=['Sample label', 'Years', 'Depth'], how='left')
df_corr_num = df_corr[corr_vars_original].dropna(axis=1, thresh=5).dropna()

corr_matrix_raw = df_corr_num.corr()

p_matrix_raw = pd.DataFrame(index=corr_matrix_raw.index, columns=corr_matrix_raw.columns, dtype=float)
for var1 in corr_matrix_raw.index:
    for var2 in corr_matrix_raw.columns:
        if var1 == var2:
            p_matrix_raw.loc[var1, var2] = 0.0
        else:
            _, p_val = stats.pearsonr(df_corr_num[var1], df_corr_num[var2])
            p_matrix_raw.loc[var1, var2] = p_val

corr_matrix_raw.to_csv(output_dir + "correlation_matrix_r.csv")
p_matrix_raw.to_csv(output_dir + "correlation_matrix_p.csv")

common_vars = [v for v in corr_vars_original if v in corr_matrix_raw.columns]
common_labels = [var_labels[corr_vars_original.index(v)] for v in common_vars]
corr_matrix = corr_matrix_raw.loc[common_vars, common_vars]
corr_matrix.index = common_labels
corr_matrix.columns = common_labels

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
cmap = sns.diverging_palette(240, 10, as_cmap=True)
hm = sns.heatmap(corr_matrix, mask=mask, annot=True, cmap=cmap, center=0,
                 fmt='.2f', linewidths=0.5, cbar_kws={"shrink": 0.8},
                 annot_kws={"fontsize": 10})
hm.set_xticklabels(hm.get_xticklabels(), rotation=45, ha='right', fontsize=10)
hm.set_yticklabels(hm.get_yticklabels(), rotation=0, fontsize=10)
plt.title('Correlation between mineralization parameters\nand soil properties', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + "Fig4_correlation_heatmap.png", dpi=600)
plt.close()

# --- 5.5 整合点图 (Fig. 3) ---
tukey_c0 = pairwise_tukeyhsd(df_fit['C0_fast'].dropna(), df_fit['Years'].loc[df_fit['C0_fast'].notna()])
tukey_k = pairwise_tukeyhsd(df_fit['k_fast'].dropna(), df_fit['Years'].loc[df_fit['k_fast'].notna()])

def add_significance_to_axes(ax, data, var, tukey_result, order):
    rows = tukey_result._results_table.data[1:]
    sig_pairs = []
    for row in rows:
        if len(row) >= 7:
            g1, g2, meandiff, p, lower, upper, reject = row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            if str(reject) == 'True':
                sig_pairs.append((str(g1), str(g2), float(p)))
    y_max = data[var].max()
    y_min = data[var].min()
    y_range = y_max - y_min
    offset_step = y_range * 0.06
    annotated = set()
    count = 0
    for g1, g2, p in sig_pairs:
        if (g1, g2) in annotated or (g2, g1) in annotated:
            continue
        annotated.add((g1, g2))
        x1, x2 = order.index(g1), order.index(g2)
        y_line = y_max + (count + 1) * offset_step
        ax.plot([x1, x1, x2, x2], [y_line, y_line + offset_step*0.1, y_line + offset_step*0.1, y_line],
                lw=1.2, c='k')
        stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text((x1 + x2) / 2, y_line + offset_step*0.15, stars, ha='center', va='bottom', fontsize=13)
        count += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.pointplot(data=df_fit, x='Years', y='C0_fast', hue='Depth', order=year_order,
              palette='deep', ax=ax1, capsize=0.1, errwidth=1.5, markers='D', linestyles='none')
sns.stripplot(data=df_fit, x='Years', y='C0_fast', hue='Depth', order=year_order,
              palette='deep', ax=ax1, dodge=False, alpha=0.5, size=6, edgecolor='gray', linewidth=0.5)
ax1.set_ylabel('C$_{0,fast}$ (mg CO$_2$-C g$^{-1}$ SOC)')
ax1.set_title('a', loc='left', fontweight='bold')
ax1.legend_.remove()

sns.pointplot(data=df_fit, x='Years', y='k_fast', hue='Depth', order=year_order,
              palette='deep', ax=ax2, capsize=0.1, errwidth=1.5, markers='s', linestyles='none')
sns.stripplot(data=df_fit, x='Years', y='k_fast', hue='Depth', order=year_order,
              palette='deep', ax=ax2, dodge=False, alpha=0.5, size=6, edgecolor='gray', linewidth=0.5)
ax2.set_ylabel('k$_{fast}$ (d$^{-1}$)')
ax2.set_title('b', loc='left', fontweight='bold')
ax2.legend_.remove()

add_significance_to_axes(ax1, df_fit, 'C0_fast', tukey_c0, year_order)
add_significance_to_axes(ax2, df_fit, 'k_fast', tukey_k, year_order)

handles, labels = ax1.get_legend_handles_labels()
unique_labels = []
unique_handles = []
for h, l in zip(handles, labels):
    if l not in unique_labels:
        unique_labels.append(l)
        unique_handles.append(h)
fig.legend(unique_handles, unique_labels, title='Depth', loc='upper right',
           bbox_to_anchor=(1.1, 0.9), frameon=True)

for ax in (ax1, ax2):
    ax.set_xlabel('Recovery year')
fig.tight_layout()
plt.savefig(output_dir + "Fig3_integrated_pointplot.png", dpi=600)
plt.close()

# --- 5.6 k_fast vs Alkyl/O-alkyl ratio (Supplementary) ---
df_mech = df_fit.merge(df3, left_on=['SampleID', 'Years', 'Depth'],
                       right_on=['Sample label', 'Years', 'Depth'], how='left')
df_mech = df_mech.dropna(subset=['k_fast', 'Alkyl C/O-alkyl C'])

fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(data=df_mech, x='Alkyl C/O-alkyl C', y='k_fast', ax=ax,
            scatter_kws={'s': 50, 'color': '#2E86AB', 'alpha': 0.7, 'edgecolor': 'white'},
            line_kws={'color': '#A23B72', 'lw': 2})
r, p = stats.pearsonr(df_mech['Alkyl C/O-alkyl C'], df_mech['k_fast'])
ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}', transform=ax.transAxes,
        verticalalignment='top', fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
ax.set_xlabel('Alkyl C / O-alkyl C ratio', fontsize=12)
ax.set_ylabel('k$_{fast}$ (d$^{-1}$)', fontsize=12)
ax.set_title('Relationship between k$_{fast}$ and carbon quality', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + "FigS_kfast_vs_AlkylOalkyl.png", dpi=600)
plt.close()

# --- 新增 Fig. SX: 按恢复年限分组的 Organic Chemistry vs. k_fast 散点图 ---
df_oc = df_mech[['Years', 'k_fast', 'Alkyl C/O-alkyl C']].dropna().copy()
df_oc['Years'] = pd.Categorical(df_oc['Years'], categories=year_order, ordered=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
axes = axes.flatten()

for i, year in enumerate(year_order):
    ax = axes[i]
    subset = df_oc[df_oc['Years'] == year]
    if len(subset) < 3:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(year)
        continue
    x = subset['Alkyl C/O-alkyl C']
    y = subset['k_fast']
    sns.regplot(x=x, y=y, ax=ax, scatter_kws={'s': 60, 'alpha': 0.7, 'edgecolor': 'white'},
                line_kws={'color': '#A23B72', 'lw': 2})
    r, p = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}', transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
    ax.set_title(year, fontsize=13, fontweight='bold')
    ax.grid(False)

fig.supxlabel('Alkyl C / O-alkyl C ratio', fontsize=13)
fig.supylabel('k$_{fast}$ (d$^{-1}$)', fontsize=13)
fig.suptitle('Organic chemistry vs. k$_{fast}$ by recovery year', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_dir + "FigSX_kfast_vs_OrganicChemistry_byYear.png", dpi=600)
plt.close()

# --- 5.7 核心发现综合图 ---
fig, ax1 = plt.subplots(figsize=(10, 6))
color1 = '#1F78B4'
ax1.set_xlabel('Recovery year', fontsize=12)
ax1.set_ylabel('k$_{fast}$ (d$^{-1}$)', color=color1, fontsize=12)
df_k = df_fit.groupby('Years')['k_fast'].mean()
x_pos = np.arange(len(df_k))
ax1.plot(x_pos, df_k.values, 'o-', color=color1, linewidth=2.5, markersize=9,
         markerfacecolor='white', markeredgewidth=1.5)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_k.index, fontsize=10)

ax2 = ax1.twinx()
color2 = '#D62728'
df_ratio = df_mech.groupby('Years')['Alkyl C/O-alkyl C'].mean()
ax2.set_ylabel('Alkyl / O-alkyl C ratio', color=color2, fontsize=12)
ax2.plot(x_pos, df_ratio.values, 's--', color=color2, linewidth=2.5, markersize=9,
         markerfacecolor='white', markeredgewidth=1.5)
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_title('Core finding: k$_{fast}$ decline linked to increasing Alkyl/O-alkyl ratio',
              fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(output_dir + "Fig_core_finding.png", dpi=600)
plt.close()

# ======================== 6. 机制分析（修正版：使用PCA解决共线性） ========================
# ---------- 扩展变量重命名，加入新的PLFA指标 ----------
rename_dict = {
    'SOC [g/kg]': 'SOC', 'TN [g/kg]': 'TN', 'C/N': 'CN', 'DOC [mg/kg]': 'DOC',
    'O-alkyl C [%]': 'O_alkyl', 'Alkyl C [%]': 'Alkyl', 'Alkyl C/O-alkyl C': 'Alkyl_O_alkyl',
    'PLFA [µg/g] (Total)': 'TotalPLFA', 'fungalPLFA/bacterialPLFA': 'FBRatio',
    'BG [nmol/g/h]': 'BG', 'NAG [nmol/g/h]': 'NAG',
    # 新增的PLFA指标重命名
    'PLFA [µg/g] (Actinobacteria)': 'PLFA_Actino',
    'PLFA [µg/g] (fungi)': 'PLFA_fungi',
    'PLFA [µg/g] (Bacteria)': 'PLFA_Bacteria'
}
df_mech = df_mech.rename(columns=rename_dict)

chem_vars = ['O_alkyl', 'Alkyl', 'Alkyl_O_alkyl']
other_vars = ['SOC', 'TN', 'CN', 'DOC', 'TotalPLFA', 'FBRatio', 'BG', 'NAG']

df_mech_clean = df_mech.dropna(subset=chem_vars + other_vars + ['C0_fast', 'k_fast']).reset_index(drop=True)

X_chem = df_mech_clean[chem_vars].values
scaler = StandardScaler()
X_chem_scaled = scaler.fit_transform(X_chem)
pca = PCA(n_components=2)
pc_scores = pca.fit_transform(X_chem_scaled)

df_mech_clean['PC1_chem'] = pc_scores[:, 0]
df_mech_clean['PC2_chem'] = pc_scores[:, 1]

pred_vars_clean = other_vars + ['PC1_chem', 'PC2_chem']

# ======================== 6.1 PCA 载荷图（圆形双标图版，定制标签） ========================
loadings = pd.DataFrame(pca.components_.T, index=chem_vars, columns=['PC1', 'PC2'])

# 变量名显示映射
label_map = {
    'O_alkyl': 'O alkyl C',
    'Alkyl': 'Alkyl C',
    'Alkyl_O_alkyl': 'A/O'
}

fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect('equal')

# 单位圆
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--', linewidth=1.2, alpha=0.6)
ax.add_artist(circle)

# 坐标轴
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)

# 为每个变量绘制向量箭头和标签
colors = sns.color_palette("husl", len(loadings))
for i, (var, row) in enumerate(loadings.iterrows()):
    x, y = row['PC1'], row['PC2']
    ax.arrow(0, 0, x, y,
             head_width=0.05, head_length=0.05,
             fc=colors[i], ec=colors[i],
             linewidth=2.5, alpha=0.85,
             length_includes_head=True)
    display_label = label_map.get(var, var)          # 使用映射后的名称
    ax.text(x * 1.15, y * 1.15, display_label,
            ha='center', va='center',
            fontsize=12, fontweight='bold',
            color=colors[i],
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1))

# 坐标轴范围与标签
limit = max(abs(loadings.values).max() * 1.3, 1.2)
ax.set_xlim(-limit, limit)
ax.set_ylim(-limit, limit)
ax.set_xlabel('PC1', fontsize=13, fontweight='bold')
ax.set_ylabel('PC2', fontsize=13, fontweight='bold')
ax.set_title('PCA Loadings of Organic Chemistry\n(Vector Length = Variable Contribution)',
             fontsize=14, fontweight='bold', pad=18)
ax.grid(True, linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(output_dir + "FigS_PCA_loadings.png", dpi=600)
plt.close()

# --- 6.2 LMG 条形图（使用PCA得分） ---
def hierarchical_partitioning(X, y, names):
    n, p = X.shape
    total_r2 = LinearRegression().fit(X, y).score(X, y)
    lmg = np.zeros(p)
    for var_idx in range(p):
        other_vars_idx = [j for j in range(p) if j != var_idx]
        for k in range(len(other_vars_idx) + 1):
            for subset in combinations(other_vars_idx, k):
                subset_indices = list(subset)
                if len(subset_indices) == 0:
                    r2_without = 0.0
                else:
                    X_sub = X[:, subset_indices]
                    r2_without = LinearRegression().fit(X_sub, y).score(X_sub, y)
                subset_with = subset_indices + [var_idx]
                X_sub_with = X[:, subset_with]
                r2_with = LinearRegression().fit(X_sub_with, y).score(X_sub_with, y)
                increment = r2_with - r2_without
                lmg[var_idx] += increment / (len(subset) + 1)
    lmg = lmg / np.sum(lmg) * total_r2 if np.sum(lmg) > 0 else np.zeros(p)
    return pd.DataFrame({'Variable': names, 'Independent_contribution': lmg}).sort_values(
        by='Independent_contribution', ascending=True)

hp_c0 = hierarchical_partitioning(df_mech_clean[pred_vars_clean].values,
                                   df_mech_clean['C0_fast'].values,
                                   pred_vars_clean)
hp_k = hierarchical_partitioning(df_mech_clean[pred_vars_clean].values,
                                  df_mech_clean['k_fast'].values,
                                  pred_vars_clean)

label_map_lmg = {
    'SOC': 'SOC', 'TN': 'TN', 'CN': 'C/N', 'DOC': 'DOC',
    'PC1_chem': 'PC1 (Chem)', 'PC2_chem': 'PC2 (Chem)',
    'TotalPLFA': 'Total PLFA', 'FBRatio': 'F/B ratio', 'BG': 'BG', 'NAG': 'NAG'
}
hp_c0['Variable'] = hp_c0['Variable'].map(label_map_lmg)
hp_k['Variable'] = hp_k['Variable'].map(label_map_lmg)

def gradient_bars(ax, df, bar_color='Blues'):
    values = df['Independent_contribution'].values
    norm = Normalize(vmin=values.min(), vmax=values.max())
    cmap_local = plt.get_cmap(bar_color)
    colors = cmap_local(norm(values))
    ax.barh(df['Variable'], values, color=colors, edgecolor='black', linewidth=0.6)
    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=10)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
gradient_bars(axes[0], hp_k, 'Blues')
axes[0].set_xlabel('Independent contribution (R$^2$)', fontsize=12)
axes[0].set_title('k$_{fast}$', fontsize=14, fontweight='bold')
axes[0].set_xlim(0, 0.3)

gradient_bars(axes[1], hp_c0, 'Reds')
axes[1].set_xlabel('Independent contribution (R$^2$)', fontsize=12)
axes[1].set_title('C$_{0,fast}$', fontsize=14, fontweight='bold')
axes[1].set_xlim(0, 0.4)

fig.suptitle('Hierarchical partitioning of explanatory power (LMG)', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(output_dir + "Fig5_LMG_partitioning.png", dpi=600)
plt.close()

# ======================== 6.4 PLS-SEM 分析与路径图（使用PCA得分，更新微生物指标） ========================
print("\n========== PLS-SEM 分析 ==========")

from sklearn.cross_decomposition import PLSRegression
from scipy.stats import norm

# 更新潜变量定义：微生物生物量现在用五个PLFA指标
lv_list = {
    "Recovery_Time": ["Years_num"],
    "Organic_Chemistry": ["PC1_chem", "PC2_chem"],
    "Microbial_Biomass": ["PLFA_Actino", "PLFA_fungi", "PLFA_Bacteria", "TotalPLFA", "FBRatio"],
    "k_fast": ["k_fast"]
}

df_pls = df_mech_clean.copy()
years_cat = pd.Categorical(df_pls['Years'], categories=year_order, ordered=True)
df_pls['Years_num'] = years_cat.codes

# 确保所有新的微生物指标无缺失
df_pls = df_pls.dropna(subset=['PLFA_Actino', 'PLFA_fungi', 'PLFA_Bacteria', 'TotalPLFA', 'FBRatio'])

# 对应修改 all_vars，移除BG，加入新PLFA指标
all_vars = ['Years_num', 'PC1_chem', 'PC2_chem',
            'PLFA_Actino', 'PLFA_fungi', 'PLFA_Bacteria', 'TotalPLFA', 'FBRatio',
            'k_fast']
X_all = StandardScaler().fit_transform(df_pls[all_vars])
X_df = pd.DataFrame(X_all, columns=all_vars)

lv_scores = {}
for lv, mvars in lv_list.items():
    if len(mvars) == 1:
        lv_scores[lv] = X_df[mvars[0]].values.reshape(-1, 1)
    else:
        pca_inner = PCA(n_components=1)
        lv_scores[lv] = pca_inner.fit_transform(X_df[mvars])

scores_df = pd.DataFrame({name: sc.flatten() for name, sc in lv_scores.items()})

paths = [
    ("Recovery_Time", "Organic_Chemistry"),
    ("Recovery_Time", "Microbial_Biomass"),
    ("Organic_Chemistry", "Microbial_Biomass"),
    ("Recovery_Time", "k_fast"),
    ("Organic_Chemistry", "k_fast"),
    ("Microbial_Biomass", "k_fast"),
]

path_coefs = {}
for src, tgt in paths:
    model = LinearRegression().fit(scores_df[[src]].values, scores_df[tgt].values)
    path_coefs[(src, tgt)] = model.coef_[0]

n_boot = 5000
rng = np.random.default_rng(42)
boot_coefs = {p: [] for p in paths}
for _ in range(n_boot):
    idx = rng.choice(len(scores_df), size=len(scores_df), replace=True)
    boot_scores = scores_df.iloc[idx]
    for src, tgt in paths:
        X_b = boot_scores[[src]].values
        y_b = boot_scores[tgt].values
        model = LinearRegression().fit(X_b, y_b)
        boot_coefs[(src, tgt)].append(model.coef_[0])

path_stats = []
for p in paths:
    arr = np.array(boot_coefs[p])
    ci_lower, ci_upper = np.percentile(arr, [2.5, 97.5])
    se = np.std(arr, ddof=1)
    z = path_coefs[p] / se if se != 0 else 0
    p_value = 2 * (1 - norm.cdf(abs(z)))
    sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else "ns"))
    path_stats.append({
        'From': p[0], 'To': p[1],
        'Coefficient': path_coefs[p],
        'CI_lower': ci_lower, 'CI_upper': ci_upper,
        'p_value': p_value, 'Significance': sig
    })
path_stats_df = pd.DataFrame(path_stats)

fig, ax = plt.subplots(figsize=(10, 7.5))
pos = {
    "Recovery_Time": (0.15, 0.5),
    "Organic_Chemistry": (0.55, 0.82),
    "Microbial_Biomass": (0.55, 0.18),
    "k_fast": (0.9, 0.5)
}
node_colors = {
    'Recovery_Time': '#1F78B4',
    'Organic_Chemistry': '#33A02C',
    'Microbial_Biomass': '#E31A1C',
    'k_fast': '#6A3D9A'
}

for name, (x, y) in pos.items():
    ax.add_patch(mpatches.FancyBboxPatch((x-0.1, y-0.06), 0.2, 0.12,
                                         boxstyle="round,pad=0.02",
                                         linewidth=2, edgecolor='black',
                                         facecolor=node_colors[name], alpha=0.9,
                                         zorder=5))
    ax.text(x, y, name.replace('_', '\n'), ha='center', va='center',
            fontsize=10, fontweight='bold', color='white', zorder=6)

for (src, tgt) in paths:
    coef = path_coefs[(src, tgt)]
    sig_row = path_stats_df[(path_stats_df['From']==src) & (path_stats_df['To']==tgt)]
    if len(sig_row) == 0: continue
    sig = sig_row['Significance'].values[0]
    p_val = sig_row['p_value'].values[0]
    lw = 2.5 if abs(coef) >= 0.3 else 1.8
    color = '#2c3e50' if abs(coef) >= 0.3 and p_val < 0.05 else '#95a5a6'
    ls = '-' if p_val < 0.05 else '--'
    x1, y1 = pos[src]
    x2, y2 = pos[tgt]

    if src == "Recovery_Time" and tgt == "k_fast":
        connectionstyle = "arc3,rad=0"
    elif src == "Recovery_Time":
        connectionstyle = "arc3,rad=0.2"
    elif src == "Organic_Chemistry" and tgt == "Microbial_Biomass":
        connectionstyle = "arc3,rad=-0.2"
    elif src == "Organic_Chemistry" and tgt == "k_fast":
        connectionstyle = "arc3,rad=0.25"
    elif src == "Microbial_Biomass" and tgt == "k_fast":
        connectionstyle = "arc3,rad=-0.25"
    else:
        connectionstyle = "arc3,rad=0"

    arrow = mpatches.FancyArrowPatch((x1, y1), (x2, y2),
                                     connectionstyle=connectionstyle,
                                     arrowstyle='-|>', color=color,
                                     lw=lw, linestyle=ls,
                                     mutation_scale=15, zorder=3)
    ax.add_patch(arrow)

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    if 'rad' in connectionstyle:
        rad = float(connectionstyle.split('=')[1])
        mid_x += rad * (y2 - y1) * 0.3
        mid_y -= rad * (x2 - x1) * 0.3
    label = f'{coef:.2f}{sig}'
    ax.text(mid_x, mid_y, label, fontsize=11,
            ha='center', va='center',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.2'),
            zorder=7)

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Path model of thaw slump effects on k$_{fast}$', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(output_dir + "Fig6_PLS_path_diagram.png", dpi=600)
plt.close()

# 中介效应图 (Fig. 7) - 基于新的微生物指标结果
indirect1 = path_coefs[("Recovery_Time", "Organic_Chemistry")] * path_coefs[("Organic_Chemistry", "k_fast")]
indirect2 = path_coefs[("Recovery_Time", "Microbial_Biomass")] * path_coefs[("Microbial_Biomass", "k_fast")]
indirect3 = path_coefs[("Recovery_Time", "Organic_Chemistry")] * path_coefs[("Organic_Chemistry", "Microbial_Biomass")] * path_coefs[("Microbial_Biomass", "k_fast")]
total_indirect = indirect1 + indirect2 + indirect3
direct = path_coefs[("Recovery_Time", "k_fast")]
total_effect = direct + total_indirect

boot_indirect = []
for i in range(n_boot):
    arr = {p: boot_coefs[p][i] for p in paths}
    ind = (arr[("Recovery_Time", "Organic_Chemistry")] * arr[("Organic_Chemistry", "k_fast")] +
           arr[("Recovery_Time", "Microbial_Biomass")] * arr[("Microbial_Biomass", "k_fast")] +
           arr[("Recovery_Time", "Organic_Chemistry")] * arr[("Organic_Chemistry", "Microbial_Biomass")] * arr[("Microbial_Biomass", "k_fast")])
    boot_indirect.append(ind)
ind_ci_low, ind_ci_up = np.percentile(boot_indirect, [2.5, 97.5])

effects = ['Direct', 'Indirect\n(via Organic\nChemistry)', 'Indirect\n(via Microbial\nBiomass)',
           'Indirect\n(serial)', 'Total\nIndirect']
values = [direct, indirect1, indirect2, indirect3, total_indirect]

fig, ax = plt.subplots(figsize=(8, 5))
colors_bar = ['#95a5a6', '#1F78B4', '#1F78B4', '#1F78B4', '#E31A1C']
bars = ax.bar(effects, values, color=colors_bar, edgecolor='white', linewidth=1.2)
ax.errorbar(4, values[4], yerr=[[values[4]-ind_ci_low], [ind_ci_up-values[4]]],
            fmt='none', ecolor='black', capsize=6, capthick=1.5)
ax.set_ylabel('Effect size (standardized)', fontsize=12)
ax.set_title('Mediation effects of recovery time on k$_{fast}$', fontsize=14, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
for i, v in enumerate(values):
    ax.text(i, v + 0.02*np.sign(v) if v != 0 else 0.02, f'{v:.3f}',
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(output_dir + "Fig7_mediation_effects.png", dpi=600)
plt.close()

path_stats_df.to_csv(output_dir + "PLS_bootstrap_results.csv", index=False)
effect_table = pd.DataFrame({
    'Effect': ['Direct', 'Indirect (via Organic_Chemistry)', 'Indirect (via Microbial_Biomass)',
               'Indirect (serial)', 'Total Indirect', 'Total Effect'],
    'Coefficient': [direct, indirect1, indirect2, indirect3, total_indirect, total_effect],
    'CI_lower': [np.nan]*4 + [ind_ci_low, np.nan],
    'CI_upper': [np.nan]*4 + [ind_ci_up, np.nan]
})
effect_table.to_csv(output_dir + "PLS_mediation_effects.csv", index=False)

# ======================== 7. 描述性统计 (Table 1) ========================
soil_vars = ['SOC [g/kg]', 'TN [g/kg]', 'C/N', 'DOC [mg/kg]',
             'O-alkyl C [%]', 'Alkyl C [%]', 'Alkyl C/O-alkyl C',
             'PLFA [µg/g] (Total)', 'fungalPLFA/bacterialPLFA', 'BG [nmol/g/h]', 'NAG [nmol/g/h]']
available_vars = [v for v in soil_vars if v in df1.columns]
table1 = df1.groupby(['Years', 'Depth'])[available_vars].agg(['mean', 'std']).round(2)
table1.to_csv(output_dir + "Table1_soil_properties.csv")

desc_stats = df_fit.groupby(['Years', 'Depth']).agg({
    'C0_total': ['mean', 'std', 'count'],
    'C0_fast': ['mean', 'std'],
    'k_fast': ['mean', 'std'],
}).round(3)
desc_stats.to_csv(output_dir + "descriptive_stats.csv")

print("\n所有分析完成！EST 期刊风格图表均已输出至：", output_dir)
print("提示：Fig. 1 研究区与样地设计请手动制作。")
print("新增 Fig. SX：按恢复年限分组的 Organic Chemistry vs. k_fast 散点图已生成。")

# ======================== 8. 参数上限敏感性分析 ========================
print("\n========== 开始参数上限敏感性分析 ==========")

k_fast_ubs = [0.5, 1.0, 2.0, 5.0, 1e6]
k_slow_ubs = [0.2, 0.5, 1.0, 2.0]

combinations_to_test = []
for kf in k_fast_ubs:
    combinations_to_test.append((kf, 0.5))
for ks in k_slow_ubs:
    if (1.0, ks) not in combinations_to_test:
        combinations_to_test.append((1.0, ks))

combinations_to_test = list(set(combinations_to_test))
print(f"将测试 {len(combinations_to_test)} 组参数上限组合")

def fit_double_with_bounds(sample_data, k_fast_ub, k_slow_ub):
    t = sample_data['Day'].values
    y = sample_data['CO2-C/SOC aerob [mg/g]'].values
    n = len(t)
    if n < 4:
        return [np.nan]*4
    C0_init = y.max() * 1.1
    p0 = [C0_init*0.3, 0.2, C0_init*0.7, 0.01]
    try:
        popt, _ = curve_fit(double_pool, t, y, p0=p0,
                            bounds=(0, [np.inf, k_fast_ub, np.inf, k_slow_ub]),
                            maxfev=10000)
        return popt
    except:
        return [np.nan]*4

sample_data_dict = {}
for sid in sample_ids:
    sub = df1[df1['SampleID'] == sid].copy()
    sub_data = sub[['Day', 'CO2-C/SOC aerob [mg/g]']].dropna()
    if len(sub_data) >= 4:
        sample_data_dict[sid] = sub_data

sens_results = []
for k_fast_ub, k_slow_ub in combinations_to_test:
    row_data = {'k_fast_ub': k_fast_ub, 'k_slow_ub': k_slow_ub}
    for sid, sdata in sample_data_dict.items():
        params = fit_double_with_bounds(sdata, k_fast_ub, k_slow_ub)
        row_data[sid + '_C0_fast'] = params[0]
        row_data[sid + '_k_fast'] = params[1]
        row_data[sid + '_C0_slow'] = params[2]
        row_data[sid + '_k_slow'] = params[3]
    sens_results.append(row_data)

df_sens = pd.DataFrame(sens_results)

kfast_cols = [c for c in df_sens.columns if c.endswith('_k_fast')]
id_vars = ['k_fast_ub', 'k_slow_ub']
df_kfast_long = df_sens.melt(id_vars=id_vars, value_vars=kfast_cols,
                             var_name='SampleID_temp', value_name='k_fast')
df_kfast_long['SampleID'] = df_kfast_long['SampleID_temp'].str.replace('_k_fast', '', regex=False)
df_kfast_long = df_kfast_long.dropna(subset=['k_fast'])

kslow_cols = [c for c in df_sens.columns if c.endswith('_k_slow')]
df_kslow_long = df_sens.melt(id_vars=id_vars, value_vars=kslow_cols,
                             var_name='SampleID_temp', value_name='k_slow')
df_kslow_long['SampleID'] = df_kslow_long['SampleID_temp'].str.replace('_k_slow', '', regex=False)
df_kslow_long = df_kslow_long.dropna(subset=['k_slow'])

df_kfast_fixed_slow = df_kfast_long[df_kfast_long['k_slow_ub'] == 0.5].copy()
order_kf = sorted(df_kfast_fixed_slow['k_fast_ub'].unique())
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_kfast_fixed_slow, x='k_fast_ub', y='k_fast', order=order_kf,
            palette='Blues')
plt.xlabel('Upper bound of $k_{fast}$ (d$^{-1}$)')
plt.ylabel('Fitted $k_{fast}$ (d$^{-1}$)')
plt.title('Sensitivity of $k_{fast}$ to its upper bound (with $k_{slow}$ bound = 0.5)')
plt.axhline(y=1.0, color='red', linestyle='--', label='Original bound (1.0)')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + "sensitivity_kfast_boxplot.png", dpi=600)
plt.close()

df_kslow_fixed_fast = df_kslow_long[df_kslow_long['k_fast_ub'] == 1.0].copy()
order_ks = sorted(df_kslow_fixed_fast['k_slow_ub'].unique())
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_kslow_fixed_fast, x='k_slow_ub', y='k_slow', order=order_ks,
            palette='Reds')
plt.xlabel('Upper bound of $k_{slow}$ (d$^{-1}$)')
plt.ylabel('Fitted $k_{slow}$ (d$^{-1}$)')
plt.title('Sensitivity of $k_{slow}$ to its upper bound (with $k_{fast}$ bound = 1.0)')
plt.axhline(y=0.5, color='blue', linestyle='--', label='Original bound (0.5)')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + "sensitivity_kslow_boxplot.png", dpi=600)
plt.close()

def calc_hit_rate(df_long, param_name, ub_name):
    df = df_long.copy()
    df['at_bound'] = np.abs(df[param_name] - df[ub_name]) < 1e-4
    hit_summary = df.groupby(ub_name)['at_bound'].mean().reset_index()
    hit_summary.columns = [ub_name, 'fraction_at_bound']
    return hit_summary

hit_kfast = calc_hit_rate(df_kfast_fixed_slow, 'k_fast', 'k_fast_ub')
hit_kslow = calc_hit_rate(df_kslow_fixed_fast, 'k_slow', 'k_slow_ub')

hit_kfast.to_csv(output_dir + "sensitivity_kfast_hit_rate.csv", index=False)
hit_kslow.to_csv(output_dir + "sensitivity_kslow_hit_rate.csv", index=False)

print("参数上限敏感性分析完成，输出文件：")
print("- sensitivity_kfast_boxplot.png")
print("- sensitivity_kslow_boxplot.png")
print("- sensitivity_kfast_hit_rate.csv")
print("- sensitivity_kslow_hit_rate.csv")

print("\n===== 敏感性分析关键结论 =====")
print("k_fast 触及上限比例：")
print(hit_kfast)
print("\nk_slow 触及上限比例：")
print(hit_kslow)
print("若原上限（k_fast=1.0, k_slow=0.5）下触及率极低，则表明上限对参数估计不构成实质约束。")
print("================================\n")

# ======================== 8.1 结论稳定性检验 (k_fast) ========================
print("\n========== 开始结论稳定性检验 ==========")

sample_meta = df_fit[['SampleID', 'Years', 'Depth']].dropna().drop_duplicates()
df_kfast_meta = df_kfast_fixed_slow.merge(sample_meta, on='SampleID', how='left')
df_kfast_meta = df_kfast_meta.dropna(subset=['Years', 'Depth'])

bounds_plot = [0.5, 1.0, 2.0, 5.0]
df_plot_stab = df_kfast_meta[df_kfast_meta['k_fast_ub'].isin(bounds_plot)].copy()
df_plot_stab['k_fast_ub'] = df_plot_stab['k_fast_ub'].astype(float)

g = sns.catplot(
    data=df_plot_stab,
    x='Years', y='k_fast',
    hue='Depth',
    col='k_fast_ub', col_wrap=2,
    kind='point', join=True, capsize=0.1, errwidth=1.5,
    palette='deep',
    height=4, aspect=1.2,
    col_order=bounds_plot
)
g.set_axis_labels('Recovery year', r'$k_{fast}$ (d$^{-1}$)')
g.set_titles('Upper bound = {col_name} d$^{{-1}}$')
g.fig.suptitle('Stability of $k_{fast}$ across upper bounds', fontsize=15, fontweight='bold', y=1.02)
g.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig(output_dir + "stability_kfast_treatment_means.png", dpi=600)
plt.close()

anova_stab_results = []
for ub in bounds_plot:
    sub = df_plot_stab[df_plot_stab['k_fast_ub'] == ub].copy()
    if len(sub) < 5:
        anova_stab_results.append({
            'k_fast_ub': ub,
            'Years_p': np.nan,
            'Depth_p': np.nan,
            'Years:Depth_p': np.nan,
            'N_samples': len(sub)
        })
        continue
    try:
        formula = 'k_fast ~ C(Years) + C(Depth) + C(Years):C(Depth)'
        model = ols(formula, data=sub).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_stab_results.append({
            'k_fast_ub': ub,
            'Years_p': anova_table.loc['C(Years)', 'PR(>F)'],
            'Depth_p': anova_table.loc['C(Depth)', 'PR(>F)'],
            'Years:Depth_p': anova_table.loc['C(Years):C(Depth)', 'PR(>F)'],
            'N_samples': len(sub)
        })
    except Exception as e:
        print(f"ANOVA failed for k_fast_ub={ub}: {e}")
        anova_stab_results.append({
            'k_fast_ub': ub,
            'Years_p': np.nan,
            'Depth_p': np.nan,
            'Years:Depth_p': np.nan,
            'N_samples': len(sub)
        })

df_anova_stab = pd.DataFrame(anova_stab_results)
df_anova_stab.to_csv(output_dir + "stability_ANOVA_kfast.csv", index=False)

print("\n===== 各上限下 ANOVA 显著性（k_fast） =====")
print(df_anova_stab.to_string(index=False))

years_sig_consistent = df_anova_stab['Years_p'].notna().all() and (df_anova_stab['Years_p'] < 0.05).all()
print(f"\nYears 效应在所有上限下均显著 (<0.05): {years_sig_consistent}")
print("若为 True，则核心结论（恢复年限显著影响 k_fast）完全不受上限选择影响。")
print("=============================================\n")

X_k = scores_df[["Recovery_Time", "Organic_Chemistry", "Microbial_Biomass"]].values
y_k = scores_df["k_fast"].values
model_k = LinearRegression().fit(X_k, y_k)
R2_kfast = model_k.score(X_k, y_k)
print(f"R²(k_fast) = {R2_kfast:.4f}  →  {R2_kfast*100:.1f}%")
# ======================== 9. 敏感性分析：排除单库样本的影响 ========================
print("\n" + "="*60)
print("========== 敏感性分析：排除单库样本（1-HT3-A1）的影响 ==========")
print("="*60)

# 准备两个数据集：包含单库（原始）和仅双库
df_full = df_fit.copy()  # 包含所有样本（含单库）
df_double_only = df_fit[df_fit['Best_model'] == 'double'].copy()  # 仅双库样本

print(f"样本数量对比：")
print(f"  包含所有样本：{len(df_full)} 个")
print(f"  仅双库样本：{len(df_double_only)} 个")
print(f"  排除的单库样本：{len(df_full) - len(df_double_only)} 个")

# ---------- 9.1 比较ANOVA结果（Years对k_fast的影响） ----------
print("\n" + "-"*60)
print("9.1 ANOVA结果对比（Years对k_fast的影响）：")
print("-"*60)

def run_anova_for_comparison(data, name):
    """运行ANOVA并返回关键p值"""
    if 'k_fast' not in data.columns or len(data) < 5:
        return {"Years_p": np.nan, "Depth_p": np.nan, "Interaction_p": np.nan}
    
    sub_data = data.dropna(subset=['k_fast'])
    if len(sub_data) < 3:
        return {"Years_p": np.nan, "Depth_p": np.nan, "Interaction_p": np.nan}
    
    formula = 'k_fast ~ C(Years) + C(Depth) + C(Years):C(Depth)'
    try:
        model = ols(formula, data=sub_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        return {
            'Years_p': anova_table.loc['C(Years)', 'PR(>F)'],
            'Depth_p': anova_table.loc['C(Depth)', 'PR(>F)'],
            'Interaction_p': anova_table.loc['C(Years):C(Depth)', 'PR(>F)']
        }
    except:
        return {"Years_p": np.nan, "Depth_p": np.nan, "Interaction_p": np.nan}

anova_full = run_anova_for_comparison(df_full, "包含所有样本")
anova_double = run_anova_for_comparison(df_double_only, "仅双库样本")

print(f"包含所有样本：")
print(f"  Years p值: {anova_full['Years_p']:.4f}")
print(f"  Depth p值: {anova_full['Depth_p']:.4f}")
print(f"  Years×Depth p值: {anova_full['Interaction_p']:.4f}")

print(f"\n仅双库样本：")
print(f"  Years p值: {anova_double['Years_p']:.4f}")
print(f"  Depth p值: {anova_double['Depth_p']:.4f}")
print(f"  Years×Depth p值: {anova_double['Interaction_p']:.4f}")

# 判断显著性是否一致
years_sig_full = anova_full['Years_p'] < 0.05
years_sig_double = anova_double['Years_p'] < 0.05
print(f"\nYears效应显著性一致性：{'✓ 一致' if years_sig_full == years_sig_double else '✗ 不一致'}")

# ---------- 9.2 比较k_fast与Alkyl/O-alkyl比的相关性 ----------
print("\n" + "-"*60)
print("9.2 相关性分析对比（k_fast vs Alkyl/O-alkyl比）：")
print("-"*60)

def calc_correlation_for_comparison(data, name):
    """计算k_fast与Alkyl/O-alkyl比的相关性"""
    merged = data.merge(df3, left_on=['SampleID', 'Years', 'Depth'],
                        right_on=['Sample label', 'Years', 'Depth'], how='left')
    merged = merged.dropna(subset=['k_fast', 'Alkyl C/O-alkyl C'])
    
    if len(merged) < 3:
        return {"r": np.nan, "p": np.nan, "n": 0}
    
    r, p = stats.pearsonr(merged['Alkyl C/O-alkyl C'], merged['k_fast'])
    return {"r": r, "p": p, "n": len(merged)}

corr_full = calc_correlation_for_comparison(df_full, "包含所有样本")
corr_double = calc_correlation_for_comparison(df_double_only, "仅双库样本")

print(f"包含所有样本（n={corr_full['n']}）：")
print(f"  Pearson r = {corr_full['r']:.3f}, p = {corr_full['p']:.4f}")

print(f"\n仅双库样本（n={corr_double['n']}）：")
print(f"  Pearson r = {corr_double['r']:.3f}, p = {corr_double['p']:.4f}")

# 判断相关性是否一致显著
corr_sig_full = corr_full['p'] < 0.05
corr_sig_double = corr_double['p'] < 0.05
print(f"\n相关性显著性一致性：{'✓ 一致' if corr_sig_full == corr_sig_double else '✗ 不一致'}")

# ---------- 9.3 比较LMG层次方差分解结果 ----------
print("\n" + "-"*60)
print("9.3 LMG层次方差分解对比（k_fast的驱动因子）：")
print("-"*60)

def run_lmg_for_comparison(data, name):
    """运行LMG分析并返回有机化学品质的贡献"""
    # 合并数据
    merged = data.merge(df3, left_on=['SampleID', 'Years', 'Depth'],
                        right_on=['Sample label', 'Years', 'Depth'], how='left')
    
    # 准备变量
    chem_vars = ['O-alkyl C [%]', 'Alkyl C [%]', 'Alkyl C/O-alkyl C']
    other_vars = ['SOC [g/kg]', 'TN [g/kg]', 'C/N', 'DOC [mg/kg]',
                  'PLFA [µg/g] (Total)', 'fungalPLFA/bacterialPLFA',
                  'BG [nmol/g/h]', 'NAG [nmol/g/h]']
    
    # 检查数据完整性
    required_vars = chem_vars + other_vars + ['k_fast']
    merged_clean = merged.dropna(subset=required_vars).reset_index(drop=True)
    
    if len(merged_clean) < 5:
        return {"PC1_contrib": np.nan, "PC2_contrib": np.nan, "Total_chem_contrib": np.nan}
    
    # PCA处理化学变量
    X_chem = merged_clean[chem_vars].values
    scaler = StandardScaler()
    X_chem_scaled = scaler.fit_transform(X_chem)
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(X_chem_scaled)
    merged_clean['PC1_chem'] = pc_scores[:, 0]
    merged_clean['PC2_chem'] = pc_scores[:, 1]
    
    # 准备预测变量
    pred_vars = other_vars + ['PC1_chem', 'PC2_chem']
    X = merged_clean[pred_vars].values
    y = merged_clean['k_fast'].values
    
    # 运行LMG
    def hierarchical_partitioning(X, y, names):
        n, p = X.shape
        total_r2 = LinearRegression().fit(X, y).score(X, y)
        lmg = np.zeros(p)
        for var_idx in range(p):
            other_vars_idx = [j for j in range(p) if j != var_idx]
            for k in range(len(other_vars_idx) + 1):
                for subset in combinations(other_vars_idx, k):
                    subset_indices = list(subset)
                    if len(subset_indices) == 0:
                        r2_without = 0.0
                    else:
                        X_sub = X[:, subset_indices]
                        r2_without = LinearRegression().fit(X_sub, y).score(X_sub, y)
                    subset_with = subset_indices + [var_idx]
                    X_sub_with = X[:, subset_with]
                    r2_with = LinearRegression().fit(X_sub_with, y).score(X_sub_with, y)
                    increment = r2_with - r2_without
                    lmg[var_idx] += increment / (len(subset) + 1)
        lmg = lmg / np.sum(lmg) * total_r2 if np.sum(lmg) > 0 else np.zeros(p)
        return pd.DataFrame({'Variable': names, 'Independent_contribution': lmg})
    
    hp_result = hierarchical_partitioning(X, y, pred_vars)
    
    # 提取有机化学品质的贡献
    pc1_contrib = hp_result[hp_result['Variable'] == 'PC1_chem']['Independent_contribution'].values[0]
    pc2_contrib = hp_result[hp_result['Variable'] == 'PC2_chem']['Independent_contribution'].values[0]
    total_chem_contrib = pc1_contrib + pc2_contrib
    
    return {
        "PC1_contrib": pc1_contrib,
        "PC2_contrib": pc2_contrib,
        "Total_chem_contrib": total_chem_contrib,
        "n_samples": len(merged_clean)
    }

lmg_full = run_lmg_for_comparison(df_full, "包含所有样本")
lmg_double = run_lmg_for_comparison(df_double_only, "仅双库样本")

print(f"包含所有样本（n={lmg_full['n_samples']}）：")
print(f"  有机化学品质总贡献（PC1+PC2）：{lmg_full['Total_chem_contrib']:.3f}")
print(f"  PC1贡献：{lmg_full['PC1_contrib']:.3f}")
print(f"  PC2贡献：{lmg_full['PC2_contrib']:.3f}")

print(f"\n仅双库样本（n={lmg_double['n_samples']}）：")
print(f"  有机化学品质总贡献（PC1+PC2）：{lmg_double['Total_chem_contrib']:.3f}")
print(f"  PC1贡献：{lmg_double['PC1_contrib']:.3f}")
print(f"  PC2贡献：{lmg_double['PC2_contrib']:.3f}")

# 判断主导性是否一致（有机化学品质是否为最大贡献因子）
chem_dominant_full = lmg_full['Total_chem_contrib'] > 0.2  # 假设>20%为显著贡献
chem_dominant_double = lmg_double['Total_chem_contrib'] > 0.2
print(f"\n有机化学品质主导性一致性：{'✓ 一致' if chem_dominant_full == chem_dominant_double else '✗ 不一致'}")

# ---------- 9.4 比较PLS-SEM路径系数 (更新为新微生物指标) ----------
print("\n" + "-"*60)
print("9.4 PLS-SEM路径系数对比（使用新微生物指标）：")
print("-"*60)

def run_pls_sem_for_comparison(data, name):
    """运行PLS-SEM并返回关键路径系数，使用新的五PLFA指标"""
    # 合并数据
    merged = data.merge(df3, left_on=['SampleID', 'Years', 'Depth'],
                        right_on=['Sample label', 'Years', 'Depth'], how='left')
    
    # 准备变量
    chem_vars = ['O-alkyl C [%]', 'Alkyl C [%]', 'Alkyl C/O-alkyl C']
    # 微生物变量改为五个PLFA指标，移除BG
    microbial_indicators = ['PLFA [µg/g] (Actinobacteria)', 'PLFA [µg/g] (fungi)',
                            'PLFA [µg/g] (Bacteria)', 'PLFA [µg/g] (Total)',
                            'fungalPLFA/bacterialPLFA']
    other_vars = ['SOC [g/kg]', 'TN [g/kg]', 'C/N', 'DOC [mg/kg]'] + microbial_indicators + ['NAG [nmol/g/h]']
    
    required_vars = chem_vars + other_vars + ['k_fast']
    merged_clean = merged.dropna(subset=required_vars).reset_index(drop=True)
    
    if len(merged_clean) < 5:
        return {"Recovery_to_kfast": np.nan, "Chem_to_kfast": np.nan, "Recovery_to_Chem": np.nan, "n_samples": len(merged_clean)}
    
    # PCA处理化学变量
    X_chem = merged_clean[chem_vars].values
    scaler = StandardScaler()
    X_chem_scaled = scaler.fit_transform(X_chem)
    pca = PCA(n_components=2)
    pc_scores = pca.fit_transform(X_chem_scaled)
    merged_clean['PC1_chem'] = pc_scores[:, 0]
    merged_clean['PC2_chem'] = pc_scores[:, 1]
    
    # 准备PLS-SEM数据
    years_cat = pd.Categorical(merged_clean['Years'], categories=year_order, ordered=True)
    merged_clean['Years_num'] = years_cat.codes
    
    # all_vars 对应新指标（不含BG）
    all_vars = ['Years_num', 'PC1_chem', 'PC2_chem'] + microbial_indicators + ['k_fast']
    X_all = StandardScaler().fit_transform(merged_clean[all_vars])
    X_df = pd.DataFrame(X_all, columns=all_vars)
    
    # 计算潜变量得分
    lv_list = {
        "Recovery_Time": ["Years_num"],
        "Organic_Chemistry": ["PC1_chem", "PC2_chem"],
        "Microbial_Biomass": microbial_indicators,  # 使用五个PLFA指标
        "k_fast": ["k_fast"]
    }
    
    lv_scores = {}
    for lv, mvars in lv_list.items():
        if len(mvars) == 1:
            lv_scores[lv] = X_df[mvars[0]].values.reshape(-1, 1)
        else:
            pca_inner = PCA(n_components=1)
            lv_scores[lv] = pca_inner.fit_transform(X_df[mvars])
    
    scores_df = pd.DataFrame({name: sc.flatten() for name, sc in lv_scores.items()})
    
    # 计算路径系数
    paths = [
        ("Recovery_Time", "Organic_Chemistry"),
        ("Recovery_Time", "Microbial_Biomass"),
        ("Organic_Chemistry", "Microbial_Biomass"),
        ("Recovery_Time", "k_fast"),
        ("Organic_Chemistry", "k_fast"),
        ("Microbial_Biomass", "k_fast"),
    ]
    
    path_coefs = {}
    for src, tgt in paths:
        model = LinearRegression().fit(scores_df[[src]].values, scores_df[tgt].values)
        path_coefs[(src, tgt)] = model.coef_[0]
    
    return {
        "Recovery_to_kfast": path_coefs[("Recovery_Time", "k_fast")],
        "Chem_to_kfast": path_coefs[("Organic_Chemistry", "k_fast")],
        "Recovery_to_Chem": path_coefs[("Recovery_Time", "Organic_Chemistry")],
        "n_samples": len(merged_clean)
    }

pls_full = run_pls_sem_for_comparison(df_full, "包含所有样本")
pls_double = run_pls_sem_for_comparison(df_double_only, "仅双库样本")

print(f"包含所有样本（n={pls_full['n_samples']}）：")
print(f"  Recovery_Time → k_fast: {pls_full['Recovery_to_kfast']:.3f}")
print(f"  Organic_Chemistry → k_fast: {pls_full['Chem_to_kfast']:.3f}")
print(f"  Recovery_Time → Organic_Chemistry: {pls_full['Recovery_to_Chem']:.3f}")

print(f"\n仅双库样本（n={pls_double['n_samples']}）：")
print(f"  Recovery_Time → k_fast: {pls_double['Recovery_to_kfast']:.3f}")
print(f"  Organic_Chemistry → k_fast: {pls_double['Chem_to_kfast']:.3f}")
print(f"  Recovery_Time → Organic_Chemistry: {pls_double['Recovery_to_Chem']:.3f}")

# 判断路径系数符号和方向是否一致
recovery_kfast_same_sign = (pls_full['Recovery_to_kfast'] > 0) == (pls_double['Recovery_to_kfast'] > 0)
chem_kfast_same_sign = (pls_full['Chem_to_kfast'] > 0) == (pls_double['Chem_to_kfast'] > 0)
print(f"\n路径系数方向一致性：")
print(f"  Recovery_Time → k_fast: {'✓ 一致' if recovery_kfast_same_sign else '✗ 不一致'}")
print(f"  Organic_Chemistry → k_fast: {'✓ 一致' if chem_kfast_same_sign else '✗ 不一致'}")

# ---------- 9.5 综合结论 ----------
print("\n" + "="*60)
print("========== 综合结论 ==========")
print("="*60)

# 创建学术英文版的敏感性分析总结表
sensitivity_data = {
    "Analysis item": [
        "ANOVA p-value (Years for k_fast)",
        "Pearson r (k_fast vs. Alkyl/O-alkyl)",
        "p-value (k_fast vs. Alkyl/O-alkyl)",
        "LMG: Total contribution of organic chemistry quality",
        "PLS path coefficient: Recovery_Time → k_fast",
        "PLS path coefficient: Organic_Chemistry → k_fast"
    ],
    "All samples": [
        f"{anova_full['Years_p']:.4f}",
        f"{corr_full['r']:.3f}",
        f"{corr_full['p']:.4f}",
        f"{lmg_full['Total_chem_contrib']:.3f}",
        f"{pls_full['Recovery_to_kfast']:.3f}",
        f"{pls_full['Chem_to_kfast']:.3f}"
    ],
    "Double-library samples": [
        f"{anova_double['Years_p']:.4f}",
        f"{corr_double['r']:.3f}",
        f"{corr_double['p']:.4f}",
        f"{lmg_double['Total_chem_contrib']:.3f}",
        f"{pls_double['Recovery_to_kfast']:.3f}",
        f"{pls_double['Chem_to_kfast']:.3f}"
    ]
}

# 生成英文一致性判断（Yes/No）
consistency_judgments = []
for i in range(6):
    if i == 0:  # ANOVA p-value
        full_sig = float(sensitivity_data["All samples"][i]) < 0.05
        double_sig = float(sensitivity_data["Double-library samples"][i]) < 0.05
        consistency_judgments.append("Yes" if full_sig == double_sig else "No")
    elif i == 1:  # Pearson r
        full_r = float(sensitivity_data["All samples"][i])
        double_r = float(sensitivity_data["Double-library samples"][i])
        consistency_judgments.append("Yes" if (full_r > 0) == (double_r > 0) else "No")
    elif i == 2:  # p-value for correlation
        full_p = float(sensitivity_data["All samples"][i])
        double_p = float(sensitivity_data["Double-library samples"][i])
        full_sig = full_p < 0.05
        double_sig = double_p < 0.05
        consistency_judgments.append("Yes" if full_sig == double_sig else "No")
    elif i == 3:  # LMG contribution
        full_contrib = float(sensitivity_data["All samples"][i])
        double_contrib = float(sensitivity_data["Double-library samples"][i])
        consistency_judgments.append("Yes" if (full_contrib > 0.2) == (double_contrib > 0.2) else "No")
    elif i == 4:  # PLS path coefficient 1
        full_coef = float(sensitivity_data["All samples"][i])
        double_coef = float(sensitivity_data["Double-library samples"][i])
        consistency_judgments.append("Yes" if (full_coef > 0) == (double_coef > 0) else "No")
    elif i == 5:  # PLS path coefficient 2
        full_coef = float(sensitivity_data["All samples"][i])
        double_coef = float(sensitivity_data["Double-library samples"][i])
        consistency_judgments.append("Yes" if (full_coef > 0) == (double_coef > 0) else "No")

sensitivity_data["Consistent?"] = consistency_judgments

sensitivity_summary = pd.DataFrame(sensitivity_data)
sensitivity_summary.to_csv(output_dir + "sensitivity_analysis_single_vs_double.csv", index=False)
print(f"\n敏感性分析详细结果已保存至：{output_dir}sensitivity_analysis_single_vs_double.csv")

# 打印一致性汇总（保持中文显示）
print("\n敏感性分析结果汇总：")
for i, row in sensitivity_summary.iterrows():
    print(f"  {row['Analysis item']}: {row['Consistent?']}")

all_consistent = all(judgment == "Yes" for judgment in consistency_judgments)
print(f"\n总体结论：排除单库样本后，{'所有' if all_consistent else '大部分'}核心结论保持一致。")
print(f"这表明单库样本（1-HT3-A1）的排除不会扭曲本研究的核心科学结论。")

print("\n" + "="*60)
print("所有分析完成！")
print("="*60)

# ======================== 10. PLS-SEM 预测相关性验证 (Q² via k-fold cross-validation，更新微生物指标) ========================
print("\n========== PLS-SEM Q² Cross-Validation ==========")

from sklearn.model_selection import KFold

# 准备数据
cv_data = df_mech_clean.copy()
years_cat = pd.Categorical(cv_data['Years'], categories=year_order, ordered=True)
cv_data['Years_num'] = years_cat.codes

# 定义潜变量指标：微生物指标更新为五个PLFA变量
lv_indicators = {
    'Recovery_Time': ['Years_num'],
    'Organic_Chemistry': ['O_alkyl', 'Alkyl', 'Alkyl_O_alkyl'],
    'Microbial_Biomass': ['PLFA_Actino', 'PLFA_fungi', 'PLFA_Bacteria', 'TotalPLFA', 'FBRatio'],
    'k_fast': ['k_fast']
}

# 内生变量列表
endogenous = ['Organic_Chemistry', 'Microbial_Biomass', 'k_fast']

# 路径关系（用于确定每个内生变量的预测因子）
paths_in = {
    'Organic_Chemistry': ['Recovery_Time'],
    'Microbial_Biomass': ['Recovery_Time', 'Organic_Chemistry'],
    'k_fast': ['Recovery_Time', 'Organic_Chemistry', 'Microbial_Biomass']
}

def get_lv_scores(data, indicators, scaler=None, pca=None, fit=False):
    """计算潜变量得分。对于多指标潜变量，先标准化再PCA取第一主成分。"""
    if len(indicators) == 1:
        # 单一指标，直接返回该列（标准化）
        vals = data[indicators[0]].values.reshape(-1, 1)
        if fit:
            scaler = StandardScaler()
            scores = scaler.fit_transform(vals)
            return scores, scaler, None
        else:
            scores = scaler.transform(vals)
            return scores
    else:
        X = data[indicators].values
        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=1)
            scores = pca.fit_transform(X_scaled)
            return scores, scaler, pca
        else:
            X_scaled = scaler.transform(X)
            scores = pca.transform(X_scaled)
            return scores

# 交叉验证设置
k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

results_q2 = {lv: {'true': [], 'pred': [], 'ss0': []} for lv in endogenous}

for train_idx, test_idx in kf.split(cv_data):
    train_data = cv_data.iloc[train_idx].reset_index(drop=True)
    test_data = cv_data.iloc[test_idx].reset_index(drop=True)
    
    # 步骤1: 拟合所有潜变量得分转换器并计算训练集得分
    lv_scores_train = {}
    lv_scalers = {}
    lv_pcas = {}
    for lv in ['Recovery_Time', 'Organic_Chemistry', 'Microbial_Biomass', 'k_fast']:
        indicators = lv_indicators[lv]
        scores, scaler, pca = get_lv_scores(train_data, indicators, fit=True)
        lv_scores_train[lv] = scores
        lv_scalers[lv] = scaler
        lv_pcas[lv] = pca

    # 步骤2: 拟合内生变量的回归模型
    reg_models = {}
    for lv in endogenous:
        predictors = paths_in[lv]
        X = np.column_stack([lv_scores_train[p] for p in predictors])
        y = lv_scores_train[lv].ravel()
        reg_models[lv] = LinearRegression().fit(X, y)

    # 步骤3: 对测试集计算潜变量得分和预测
    lv_scores_test = {}
    for lv in ['Recovery_Time', 'Organic_Chemistry', 'Microbial_Biomass', 'k_fast']:
        indicators = lv_indicators[lv]
        scores = get_lv_scores(test_data, indicators, scaler=lv_scalers[lv], pca=lv_pcas[lv], fit=False)
        lv_scores_test[lv] = scores

    for lv in endogenous:
        predictors = paths_in[lv]
        X_test = np.column_stack([lv_scores_test[p] for p in predictors])
        y_true = lv_scores_test[lv].ravel()
        y_pred = reg_models[lv].predict(X_test)
        
        y_mean_train = np.mean(lv_scores_train[lv])
        ss0_fold = (y_true - y_mean_train) ** 2
        
        results_q2[lv]['true'].extend(y_true.tolist())
        results_q2[lv]['pred'].extend(y_pred.tolist())
        results_q2[lv]['ss0'].extend(ss0_fold.tolist())

# 计算Q²
q2_table = []
for lv in endogenous:
    press = np.sum((np.array(results_q2[lv]['true']) - np.array(results_q2[lv]['pred'])) ** 2)
    sso = np.sum(np.array(results_q2[lv]['ss0']))
    q2 = 1 - press / sso if sso != 0 else -np.inf
    q2_table.append({'Latent Variable': lv, 'Q²': q2, 'PRESS': press, 'SSO': sso})
    print(f"{lv}: Q² = {q2:.4f}")

q2_df = pd.DataFrame(q2_table)
q2_df.to_csv(output_dir + "PLS_Q2_cross_validation.csv", index=False)
print("Q² results saved to PLS_Q2_cross_validation.csv")
print("所有分析完成！")
# ======================== 11. PLS‑SEM 路径系数置换检验 (Permutation Test) ========================
print("\n========== PLS‑SEM 置换检验 (Permutation Test) ==========")
print("目的：验证路径系数是否显著偏离随机水平，增强模型稳健性证据。")

n_perm = 10000                       
rng_perm = np.random.default_rng(123)  

original_coefs = path_coefs.copy()

perm_coefs = {p: [] for p in paths}

for i in range(n_perm):
    # 随机打乱每个内生变量的得分（破坏其与预测变量的关联）
    perm_scores = scores_df.copy()
    # 对每个在 paths 中作为因变量 (tgt) 的潜变量进行打乱
    # 这里直接打乱所有内生变量列，但保留 Recovery_Time（纯外生）不动
    for tgt in ['Organic_Chemistry', 'Microbial_Biomass', 'k_fast']:
        perm_scores[tgt] = rng_perm.permutation(perm_scores[tgt].values)
    
    # 重新计算每条路径的系数
    for src, tgt in paths:
        model_perm = LinearRegression().fit(perm_scores[[src]].values, perm_scores[tgt].values)
        perm_coefs[(src, tgt)].append(model_perm.coef_[0])
    
    if (i + 1) % 1000 == 0:
        print(f"  已完成 {i+1}/{n_perm} 次置换...")

# 计算每条路径的置换 p 值（双侧）
perm_results = []
for p in paths:
    orig = original_coefs[p]
    perm_arr = np.array(perm_coefs[p])
    
    # 双侧 p 值：大于等于 |orig| 的置换系数比例
    p_val_perm = np.mean(np.abs(perm_arr) >= np.abs(orig))
    
    # 均值与标准差
    mean_perm = np.mean(perm_arr)
    std_perm = np.std(perm_arr, ddof=1)
    
    # 95% 置信区间（基于置换分布，可选）
    ci_low_perm = np.percentile(perm_arr, 2.5)
    ci_high_perm = np.percentile(perm_arr, 97.5)
    
    perm_results.append({
        'From': p[0],
        'To': p[1],
        'Original_coef': orig,
        'Permuted_mean': mean_perm,
        'Permuted_std': std_perm,
        'Permutation_p': p_val_perm,
        'Perm_CI_lower': ci_low_perm,
        'Perm_CI_upper': ci_high_perm
    })
    
    print(f"  路径 {p[0]} -> {p[1]}: 原始系数 = {orig:.4f}, "
          f"置换均值 = {mean_perm:.4f}, 置换 p = {p_val_perm:.4f}")

df_perm = pd.DataFrame(perm_results)
df_perm.to_csv(output_dir + "TableS_permutation_test.csv", index=False)
print(f"\n置换检验结果已保存至 {output_dir}TableS_permutation_test.csv")
print("若置换 p < 0.05，表明路径系数显著不同于随机预期，模型关系稳健。")
print("============================================================\n")