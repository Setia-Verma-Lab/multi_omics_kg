import pandas as pd 
import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.image import imread

# make file with just rsids

# need to get more of new endo SNPs with OR and pval matched from endo GBMI in other snakemake
def generate_rsids_file(existing_file, new_file, output_existing_file, output_new_file):
    existing_endo_df = pd.read_csv(existing_file)
    new_endo_df = pd.read_csv(new_file)

    existing_endo_df['rsid'] = existing_endo_df['rsid'].apply(lambda x: f"rs{x}")
    new_endo_df['rsid'] = new_endo_df['rsid'].apply(lambda x: f"rs{x}")
    existing_endo_df.drop(columns=['score'], inplace=True)
    new_endo_df.drop(columns=['score'], inplace=True)

    existing_endo_df.to_csv(output_existing_file, index=False, header=False, sep='\t')
    new_endo_df.to_csv(output_new_file, index=False, header=False, sep='\t')

# generate_rsids_file("/Users/ananyara/Github/multi_omics_kg/link_prediction/existing_snp_endo_scores.csv", "/Users/ananyara/Github/multi_omics_kg/link_prediction/new_snp_endo_scores.csv", 'Annotations/existing_endo_biofilter_input_snps.txt', 'Annotations/new_endo_biofilter_input_snps.txt')
# generate_rsids_file("/Users/ananyara/Github/multi_omics_kg/link_prediction/existing_snp_obesity_scores.csv", "/Users/ananyara/Github/multi_omics_kg/link_prediction/new_snp_obesity_scores.csv", 'Annotations/existing_obesity_biofilter_input_snps.txt', 'Annotations/new_obesity_biofilter_input_snps.txt')

# use lindsay's snakemake wrapper to get SNP positions for these two output files -> done
# snakemake --snakefile /project/ssverma_shared/tools/lindsay_snakemake_workflows/biofilter_wrapper/Snakefile --cores 4 Annotations/new_endo_biofilter_snps_annotations.txt --configfile config_biofilter.yaml

# for this plotting function:
# per disease: need to have rsid, chr, pos, ld_score, linkpred_score, status (existing or new), OR and pval

def generate_link_pred_figure(input_file, output_file, snp_id_1, snp_id_2, snp_id_1_file, snp_id_2_file, lead_thresh=0.95, window_kb=500, disease='Endometriosis'):

    df = pd.read_csv(input_file, sep="\t")
    df['P'] = -np.log2(1 - df['linkpred_score'])
    df['chr'] = df['chr'].astype(str)
    chrom_order = [str(i) for i in range(1,23)] + ['X','Y']
    df = df[df['chr'].isin(chrom_order)]
    df['chr'] = pd.Categorical(df['chr'], categories=chrom_order, ordered=True)

    # cumulative genomic position
    offsets, cumul = {}, 0
    for chrom in chrom_order:
        maxp = df.loc[df['chr']==chrom, 'pos'].max()
        offsets[chrom] = cumul
        cumul += maxp if not np.isnan(maxp) else 0
    df['cum_pos'] = df.apply(lambda r: r['pos'] + offsets[r['chr']], axis=1)

    # get significant new SNPs
    leads = df[(df['status']=='new') & (df['linkpred_score'] >= lead_thresh)].copy()
    win = window_kb * 1_000
    df['signal_score'] = np.nan
    for _, lead in leads.iterrows():
        c0 = lead['cum_pos']
        mask = (df['cum_pos'] >= c0 - win) & (df['cum_pos'] <= c0 + win)
        df.loc[mask, 'signal_score'] = np.nanmax([
            df.loc[mask, 'signal_score'].values,
            np.full(mask.sum(), lead['linkpred_score'])
        ], axis=0)

    print(len(leads))

    # split categories
    existing  = df[df['status']=='existing']
    signal    = df[~df['signal_score'].isna()].copy()
    other_new = df[(df['status']=='new') & (df['signal_score'].isna())]

    # proximal vs non‐proximal new snps (to exisitng sNP)
    existing_pos = existing['cum_pos'].values
    signal['proximal_existing'] = signal['cum_pos'].apply(
        lambda x: np.any(np.abs(existing_pos - x) <= win)
    )
    sig_prox = signal[signal['proximal_existing']]
    sig_non  = signal[~signal['proximal_existing']]

    fig = plt.figure(figsize=(12, 10))                # taller figure
    gs  = gridspec.GridSpec(2, 2, height_ratios=[3, 3], hspace=0.2)


    ax = fig.add_subplot(gs[0, :])  # main Manhattan plot

    # categories
    ax.scatter(sig_non['cum_pos'], sig_non['P'],
               color='blue', s=20, marker='o', label='new signal (not proximal)', zorder=2)
    ax.scatter(sig_prox['cum_pos'], sig_prox['P'],
               color='purple', s=20, marker='o', label='new signal (proximal)', zorder=3)
    ax.scatter(other_new['cum_pos'], other_new['P'],
               c=other_new['chr'].cat.codes.map(lambda i: ['darkgrey','silver'][i%2]),
               s=16, marker='o', edgecolor='none', zorder=1)
    ax.scatter(existing['cum_pos'], existing['P'],
               color='black', s=16, marker='s', label='existing', zorder=4)

    # highlight two SNP IDs
    for label, snp_id in zip(['a','b'], [snp_id_1, snp_id_2]):
        row = df[df['SNP_ID'] == snp_id]
        if not row.empty:
            x, y = row['cum_pos'].iloc[0], row['P'].iloc[0]
            ax.scatter([x], [y], facecolors='none', edgecolors='black',
                       s=150, linewidth=2, zorder=5)
            ax.text(x, y, f' {label}', fontsize=16, fontweight='bold',
                    va='bottom', ha='left', zorder=6)

    # chrom ticks
    ticks, labels = [], []
    for chrom in chrom_order:
        sub = df[df['chr']==chrom]
        if sub.empty: continue
        start = offsets[chrom]
        end   = start + sub['pos'].max()
        ticks.append((start+end)/2)
        labels.append(chrom)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=90, fontsize=12)
    ax.set_xlabel('Chromosome', fontsize=14)
    ax.set_ylabel('-log₂(1−linkpred_score)', fontsize=14)
    ax.set_title(f'{disease} Link Prediction\n(signals ≥{lead_thresh}, ±{window_kb}kb)',
                 fontsize=16)
    ax.legend(title='SNP category', title_fontsize=12, fontsize=10, loc='upper right')

    # plot the two SNP‐panel images
    for i, img_file in enumerate([snp_id_1_file, snp_id_2_file]):
        ax_img = fig.add_subplot(gs[1, i])
        img = imread(img_file)
        ax_img.imshow(img)
        ax_img.axis('off')
        # label panels “(a)” and “(b)”
        ax_img.text(0.02, 0.9, f'({chr(ord("a")+i)})',
                    transform=ax_img.transAxes, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=600)
    plt.close()

generate_link_pred_figure(
    'Annotations/endo_merged_gbmi_ld_linkpred.txt',
    'new_endo_manhattan_plot.png', snp_id_2='12_14265892', snp_id_1='2_102257925', snp_id_2_file='/Users/ananyara/Github/multi_omics_kg/link_prediction/new_links_estimates/Annotations/endo_snp_1.png', snp_id_1_file='/Users/ananyara/Github/multi_omics_kg/link_prediction/new_links_estimates/Annotations/endo_snp_2.png')

generate_link_pred_figure(
    'Annotations/obesity_merged_linkpred.txt',
    'new_obesity_manhattan_plot.png', window_kb=500, disease='Obesity', snp_id_2='11_133942404', snp_id_1='2_197022693', snp_id_1_file='/Users/ananyara/Github/multi_omics_kg/link_prediction/new_links_estimates/Annotations/snp_obesity_1.png', snp_id_2_file='/Users/ananyara/Github/multi_omics_kg/link_prediction/new_links_estimates/Annotations/snp_obesity_2.png')
# lead_thresh=0.9945,