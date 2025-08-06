# taking the validation set predictions from the classifier model, and seeing how well they correspond with the endometriosis labels
# specifically, looking at any relationship with endo stage, as well as signal from adeno only vs endo only

import pandas as pd
from scipy.stats import mannwhitneyu, gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

patient_preds = pd.read_csv("patient_predictions.csv")

patient_chart_labels = pd.read_csv("chart_review_and_EHR_labels.csv")
patient_stages = pd.read_csv("endo_chart_review_surg_stage.csv")

merged_df = pd.merge(patient_preds, patient_chart_labels, on='PMBB_ID', how='left')
merged_df = pd.merge(merged_df, patient_stages, on='PMBB_ID', how='left')
merged_df['SURG_endostage'] = merged_df['SURG_endostage'].astype(str)

# MAKING DISTTRIBUTION AND STAGING BOXPLOTS AS SUBPLOTS IN A SINGLE FIG

# chart columns
chart_cols  = [c for c in merged_df.columns if c.startswith('Chart_')]
control_col = 'Chart_Adeno_or_Endo' # the one we're comparing to statistically

# control & case scores
control_scores = merged_df.loc[merged_df[control_col]==0, 'score'].dropna()
case_scores    = {c: merged_df.loc[merged_df[c]==1, 'score'].dropna()
                  for c in chart_cols}

# sample sizes & p-values
counts = {'Controls': len(control_scores),
          **{c: len(v) for c,v in case_scores.items()}}
pvals  = {c: mannwhitneyu(scores, control_scores).pvalue
          for c,scores in case_scores.items()}

# x grid for all curves
all_scores = merged_df['score'].dropna()
x = np.linspace(all_scores.min(), all_scores.max(), 300)

# subplot set up
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# first plotting distribution 
palette      = plt.cm.tab10
control_color= 'gray'

# controls
kde_ctrl = gaussian_kde(control_scores)
ax1.plot(x, kde_ctrl(x), color=control_color,
         label=f"Controls (n={counts['Controls']})")
ax1.fill_between(x, kde_ctrl(x), color=control_color, alpha=0.3)

# cases
for i, col in enumerate(chart_cols):
    kde_case = gaussian_kde(case_scores[col])
    label     = col.replace('Chart_', '').replace('_',' ')
    color     = palette(i+1)
    ax1.plot(x, kde_case(x), color=color,
             label=f"{label} (n={counts[col]}) p={pvals[col]:.3f}")
    ax1.fill_between(x, kde_case(x), color=color, alpha=0.3)

ax1.set_xlabel('Model Score', fontsize=25)
ax1.set_ylabel('Density', fontsize=25)
ax1.set_title('Score Distributions: Controls vs. Case Subtypes',
              fontsize=25)
ax1.legend(fontsize=16, loc='upper left')
ax1.tick_params(axis='both', labelsize=25)


# boxplots by endo stage plot!
# get stage data
merged_df['SURG_endostage'] = pd.to_numeric(
    merged_df['SURG_endostage'], errors='coerce')
mdf = merged_df.dropna(subset=['SURG_endostage','score'])
stages_all = sorted(mdf['SURG_endostage'].unique())
stages     = [s for s in stages_all if s != 0.0]

data_by_stage = [mdf.loc[mdf['SURG_endostage']==s, 'score'] for s in stages]
counts_stage = {s: len(vals) for s,vals in zip(stages, data_by_stage)}

bp = ax2.boxplot(
    data_by_stage,
    widths=0.6,
    patch_artist=True,
    showfliers=False
)
for box in bp['boxes']:
    box.set_facecolor('skyblue')

# x axis things
xticks      = np.arange(1, len(stages)+1)
xticklabels = [f"{int(s)} (n={counts_stage[s]})" for s in stages]
ax2.set_xticks(xticks)
ax2.set_xticklabels(xticklabels, rotation=45, ha='right')

# put p-values below each box (compared to stage 1)
baseline = data_by_stage[stages.index(1.0)]
ymin, ymax = ax2.get_ylim()
offset     = 0.05 * (ymax - ymin)
y_text     = ymin + offset

for i, s in enumerate(stages):
    if s == 1.0: 
        continue
    p = mannwhitneyu(data_by_stage[i], baseline).pvalue
    ax2.text(
        i+1, y_text,
        f"p={p:.3f}",
        ha='center', va='top', fontsize=22
    )

ax2.set_xlabel('Endo Stage', fontsize=25)
ax2.set_ylabel('Model Score', fontsize=25)
ax2.set_title('Score Distribution by Surgical Stage', fontsize=25)
ax2.tick_params(axis='both', labelsize=25)

# save
plt.tight_layout()
plt.savefig("combined_figure.png", dpi=600, bbox_inches='tight')