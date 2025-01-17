import pandas as pd
import numpy as np

def create_loq_dataframe(compound_dict):
    """
    Creates a pivoted DataFrame for the LOQ sheet, with compounds as columns
    and their LOQ values in a single row.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.

    Returns:
    - pd.DataFrame: A pivoted DataFrame for LOQ data.
    """
    loq_data = {
        "Compound": list(compound_dict.keys()),
        "LOQ": [data.get("LOQ", None) for data in compound_dict.values()]
    }
    df_loq = pd.DataFrame(loq_data)

    # Replace invalid values (e.g., NaN, inf)
    df_loq = df_loq.replace([np.nan, np.inf, -np.inf], "N/A")
    
    # Pivot: Make compounds the columns and LOQ the values
    df_loq_pivot = pd.DataFrame([df_loq.set_index("Compound")["LOQ"].to_dict()], index=["LOQ"])
    
    return df_loq_pivot

def create_smp_concentrations_dataframe(compound_dict):
    """
    Creates a DataFrame for the Concentrations sheet.
    """
    smp_concentration_data = []
    for compound, data in compound_dict.items():
        if "Sample" in data and isinstance(data["Sample"], pd.DataFrame) and not data["Sample"].empty:
            sample_df = data["Sample"].copy()
            sample_df["Sample Name"] = sample_df["Filename"].apply(lambda x: "_".join(x.split("_")[2:-2]))
            sample_df = sample_df[sample_df["Filename"].str.contains("_Smp_")]  # Filter only 'Smp' samples
            sample_df = sample_df[["Filename", "Sample Name", "Adjusted Concentration"]]
            sample_df["Compound"] = compound
            smp_concentration_data.append(sample_df)

    if smp_concentration_data:
        df_smp_concentrations = pd.concat(smp_concentration_data, ignore_index=True)
        # Pivot using Filename
        df_smp_concentrations_wide = df_smp_concentrations.pivot(index="Filename", columns="Compound", values="Adjusted Concentration").reset_index()
        # Replace Filename with Sample Name after pivoting
        sample_name_map = df_smp_concentrations.set_index("Filename")["Sample Name"].to_dict()
        df_smp_concentrations_wide["Filename"] = df_smp_concentrations_wide["Filename"].map(sample_name_map)
        compound_order = ["Filename"] + list(compound_dict.keys())
        df_smp_concentrations_wide = df_smp_concentrations_wide.reindex(columns=compound_order).fillna("<LOQ")
    else:
        df_smp_concentrations_wide = pd.DataFrame()
    
    return df_smp_concentrations_wide

def create_dup_prec_dataframe(compound_dict):
    """
    Creates a DataFrame for the QC Duplication Precision sheet, with samples as rows
    and compounds as columns, accessing only the 'Precision' key from the Duplication Precision data.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.

    Returns:
    - pd.DataFrame: A DataFrame with samples as rows and compounds as columns for Duplication Precision.
    """
    duplication_data = []

    for compound, data in compound_dict.items():
        if "Duplication Precision" in data and isinstance(data["Duplication Precision"], pd.DataFrame) and not data["Duplication Precision"].empty:
            dup_df = data["Duplication Precision"].copy()
            if "Precision" in dup_df.columns:
                dup_df = dup_df[["Sample Name", "Precision"]]
                dup_df["Precision"] = dup_df["Precision"] * 100
                dup_df["Compound"] = compound
                duplication_data.append(dup_df)

    if duplication_data:
        # Combine all duplication data into a single DataFrame
        df_duplication = pd.concat(duplication_data, ignore_index=True)
        # Pivot to get Sample Name as rows and Compounds as columns
        df_duplication_pivot = df_duplication.pivot(index="Sample Name", columns="Compound", values="Precision")
        # Ensure the columns are in the same order as compound_dict
        compound_order = list(compound_dict.keys())
        df_duplication_pivot = df_duplication_pivot.reindex(columns=compound_order).fillna("Can't calculate precision")
        df_duplication_pivot.reset_index(inplace=True)
    else:
        df_duplication_pivot = pd.DataFrame()

    return df_duplication_pivot

def create_spike_prec_dataframe(compound_dict):
    """
    Creates a DataFrame for the QC Spike Precision sheet, with samples as rows
    and compounds as columns, accessing only the 'Precision' key from the Spike Precision data.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.

    Returns:
    - pd.DataFrame: A DataFrame with samples as rows and compounds as columns for Spike Precision.
    """
    spike_data = []

    for compound, data in compound_dict.items():
        if "Spike Precision" in data and isinstance(data["Spike Precision"], pd.DataFrame) and not data["Spike Precision"].empty:
            spike_df = data["Spike Precision"].copy()
            if "Precision" in spike_df.columns:
                spike_df = spike_df[["Sample Name", "Precision"]]
                spike_df["Precision"] = spike_df["Precision"] * 100  # Scale by 100
                spike_df["Compound"] = compound
                spike_data.append(spike_df)

    if spike_data:
        # Combine all spike data into a single DataFrame
        df_spike = pd.concat(spike_data, ignore_index=True)
        # Pivot to get Sample Name as rows and Compounds as columns
        df_spike_pivot = df_spike.pivot(index="Sample Name", columns="Compound", values="Precision")
        # Ensure the columns are in the same order as compound_dict
        compound_order = list(compound_dict.keys())
        df_spike_pivot = df_spike_pivot.reindex(columns=compound_order).fillna("Can't calculate precision")
        df_spike_pivot.reset_index(inplace=True)
    else:
        df_spike_pivot = pd.DataFrame()

    return df_spike_pivot

def create_qc_prec_dataframe(compound_dict):
    """
    Creates a DataFrame for the QC Precision sheet, with samples as rows
    and compounds as columns, accessing only the 'Precision' key from the QC Precision data.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.

    Returns:
    - pd.DataFrame: A DataFrame with samples as rows and compounds as columns for QC Precision.
    """
    qc_data = []

    for compound, data in compound_dict.items():
        if "QC Precision" in data and isinstance(data["QC Precision"], pd.DataFrame) and not data["QC Precision"].empty:
            qc_df = data["QC Precision"].copy()
            if "Precision" in qc_df.columns:
                qc_df = qc_df[["Calibration Identifier", "Precision"]]
                qc_df["Precision"] = qc_df["Precision"] * 100  # Scale by 100
                qc_df["Compound"] = compound
                qc_data.append(qc_df)

    if qc_data:
        # Combine all QC data into a single DataFrame
        df_qc = pd.concat(qc_data, ignore_index=True)
        # Pivot to get Sample Name as rows and Compounds as columns
        df_qc_pivot = df_qc.pivot(index="Calibration Identifier", columns="Compound", values="Precision")
        # Ensure the columns are in the same order as compound_dict
        compound_order = list(compound_dict.keys())
        pd.set_option("future.no_silent_downcasting", True)
        df_qc_pivot = df_qc_pivot.reindex(columns=compound_order).fillna("Can't calculate precision")
        df_qc_pivot.reset_index(inplace=True)
    else:
        df_qc_pivot = pd.DataFrame()

    return df_qc_pivot


