import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    # identifyt quasi identifiers columns
    # aka sharewd columns excluding IDs/names
    common_cols = list(set(anon_df.columns).intersection(set(aux_df.columns)))
    quasi_cols = [ c for c in common_cols if c not in ["anon_id", "name"] ]

    # merge datasets on q identifiers
    merged = pd.merge(anon_df, aux_df, on=quasi_cols, how="inner")

    # count matches per anoymized record
    counts = merged.groupby("anon_id").size()

    # keep only uniquely matched anon_ids
    unique_ids = counts[counts == 1].index
    unique_matches = merged[merged["anon_id"].isin(unique_ids)]

    # return the required columns
    result = unique_matches[["anon_id", "name"]].rename(columns={"name": "matched_name"})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    num_matches = len(matches_df)
    total_anon = len(anon_df)

    return num_matches / total_anon
