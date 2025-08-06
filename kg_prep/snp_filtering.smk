# USAGE:
# snakemake -s snp_filtering.smk -c1 --use-envmodules all --profile lsf

from os.path import join
from glob import glob
from snakemake.io import glob_wildcards

PMBB_PGEN_DIR = '/chunked_pgen_files'
BIOFILTER_DIR      = '/Multi_Omics_KG/kg_prep/data/tmp'
MERGED_AFREQ_DIR   = 'data/maf_10/afreq_chr'
TMP_MAF_DIR        = 'data/maf_10/filtered_biofilter'

# discover every original *.pgen once
CHUNKS = glob_wildcards(join(PMBB_PGEN_DIR, "{chunk}.pgen")).chunk
CHRS   = [*(map(str, range(1, 23))), "X"]  

rule all:                                              
    input:
        # expand("data/maf_10/afreq_chunks/{chunk}.afreq", chunk=CHUNKS), # ran first
        # expand("data/maf_10/afreq_chr/chr{chr}.merged.afreq",   chr=CHRS) # ran second
        # expand(f"{TMP_MAF_DIR}/snp_biofilter_dbsnp_chr{{chr}}.csv.gz", chr=[c.lower() for c in CHRS]) # ran third
        f"{TMP_MAF_DIR}/snp_biofilter.csv.gz" # ran fourth (skipping chrY bc that's not in imputed data)


# ────────────────────────── rule 1 ──────────────────────────
rule maf_filter:
    input:
        pgen = lambda wc: join(PMBB_PGEN_DIR, f"{wc.chunk}.pgen"),
        pvar = lambda wc: join(PMBB_PGEN_DIR, f"{wc.chunk}.pvar"),
        psam = lambda wc: join(PMBB_PGEN_DIR, f"{wc.chunk}.psam")
    output:
        "work/maf_10/{chunk}.pgen"
    # ── 2. prefix (no extension) to feed plink2  ─────────────
    params:
        prefix = lambda wc: join(PMBB_PGEN_DIR, wc.chunk)
    shell:
        """
        mkdir -p work/maf_10
        plink2 --pfile {params.prefix} \
               --maf 0.1 \
               --make-pgen \
               --out  work/maf_10/{wildcards.chunk}
        """

# ────────────────────────── rule 2 ──────────────────────────
rule afreq:
    input:
        "work/maf_10/{chunk}.pgen"
    output:
        "data/maf_10/afreq_chunks/{chunk}.afreq"
    envmodules: 'plink/2.0-20240804'
    shell:
        """
        mkdir -p data/maf_10/afreq_chunks
        plink2 --pfile work/maf_10/{wildcards.chunk} \
               --freq \
               --out data/maf_10/afreq_chunks/{wildcards.chunk}
        """

# uncomment this AFTER the above two rules have compeltely run and finished,
# for some reason snakemake doesnt wait until all those files have finished to start this 
# so it crashes the workflow if you run it before

# ─── NEW: merge all chunk-level *.afreq for one chromosome ──────────────
rule merge_afreq_by_chr:
    input:
        lambda wc: sorted(
            glob(f"data/maf_10/afreq_chunks/*chr{wc.chr}_*.afreq")
        )
    output:
        "data/maf_10/afreq_chr/chr{chr}.merged.afreq"
    shell:
        r"""
        mkdir -p $(dirname {output})

        # first file gives us the header
        head -n 1 {input[0]} > {output}

        # append all bodies (skip their headers)
        for f in {input}; do
            tail -n +2 "$f" >> {output}
        done
        """

# after merging by chr, I added in this filtering on biofilter rule

rule filter_biofilter:
    input:
        biofilter = lambda wc: join(BIOFILTER_DIR, f"snp_biofilter_dbsnp_chr{wc.chr}.csv.gz"),
        afreq     = lambda wc: join(MERGED_AFREQ_DIR,   f"chr{wc.chr.upper()}.merged.afreq")
    output:
        filtered = f"{TMP_MAF_DIR}/snp_biofilter_dbsnp_chr{{chr}}.csv.gz"
    run:
        import pandas as pd, gzip, os, re, pathlib

        pathlib.Path(TMP_MAF_DIR).mkdir(parents=True, exist_ok=True)

        # Load the merged afreq table ─────────────────────────────────────────
        afreq_df = pd.read_csv(input.afreq, delim_whitespace=True, dtype=str)

        # Extract chr & pos from IDs that look like "chr3_12345_A_T"
        afreq_df[["chr", "pos"]] = afreq_df["ID"].str.extract(r"chr(\w+)_(\d+)")
        chr_map = {"x": "23"}           # PLINK → dbSNP mapping
        afreq_df["chr"] = afreq_df["chr"].str.lower().replace(chr_map)
        afreq_df["snp_id"] = afreq_df["chr"] + "_" + afreq_df["pos"]
        keep = set(afreq_df["snp_id"])

        # Stream-filter the large Biofilter CSV, 100 k rows at a time ─────────
        before = after = 0
        out_chunks = []
        for chunk in pd.read_csv(input.biofilter, compression="gzip", chunksize=100_000):
            before += len(chunk)
            chunk = chunk[chunk["snp_id"].astype(str).isin(keep)]
            after += len(chunk)
            out_chunks.append(chunk)

        print(f"[chr{wildcards.chr}] kept {after:,d} / {before:,d} variants")

        pd.concat(out_chunks).to_csv(output.filtered,
                                     index=False, compression="gzip")

# after filtering the dbsnp data on afreq files (above), I added in this rule to merge everything
rule merge_snp_node_biofilter:
    input:
        expand(f"{TMP_MAF_DIR}/snp_biofilter_dbsnp_chr{{chr}}.csv.gz",
               chr=[c.lower() for c in CHRS])
    output:
        filename = f"{TMP_MAF_DIR}/snp_biofilter.csv.gz"
    run:
        import pandas as pd, gzip, os
        os.makedirs(os.path.dirname(output.filename), exist_ok=True)

        header_written = False
        with gzip.open(output.filename, "wt") as fout:
            for infile in input:
                for chunk in pd.read_csv(infile, compression="gzip", chunksize=100000):
                    chunk.to_csv(fout, index=False, header=not header_written)
                    header_written = True