import pandas as pd
import numpy as np
def create_compound_summary(compound_dict, output_file, loq_sheet_name="LOQ", concentration_sheet_name="Concentrations", qc_sheet_name="QC"):
    """
    Creates an Excel file with:
    1. A LOQ sheet containing Compound Names and LOQ values.
    2. A concentrations sheet showing Adjusted Concentrations for each compound at each sample.
    3. A QC sheet showing Duplication Precision and Spike Precision results sequentially.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.
    - output_file (str): Path to the output Excel file.
    - loq_sheet_name (str): Name of the LOQ sheet.
    - concentration_sheet_name (str): Name of the concentrations sheet.
    - qc_sheet_name (str): Name of the QC sheet.
    """
    import pandas as pd
    import numpy as np

    # ------------------ Filter Compounds ------------------
    filtered_compounds = {
        compound: data
        for compound, data in compound_dict.items()
        if not ("13C2" in compound or compound.startswith("M") and compound[1].isdigit() or compound.startswith("d") and compound[1].isdigit() or compound == "PFHxS(O18)2")
    }

    print(f"Filtered compounds: {list(filtered_compounds.keys())}")

    # ------------------ LOQ Sheet Data ------------------
    loq_data = {
        "Compound": list(filtered_compounds.keys()),
        "LOQ": [data.get("LOQ", None) for data in filtered_compounds.values()]
    }
    df_loq = pd.DataFrame(loq_data)
    df_loq = df_loq.replace([np.nan, np.inf, -np.inf], None)  # Replace NaN, inf, -inf

    # ------------------ Concentrations Sheet Data ------------------
    concentration_data = []
    compound_order = list(filtered_compounds.keys())
    for compound, data in filtered_compounds.items():
        if "Sample" in data and isinstance(data["Sample"], pd.DataFrame) and not data["Sample"].empty:
            sample_df = data["Sample"].copy()
            sample_df["Sample Type"] = sample_df["Filename"].str.split("_").str[1]
            sample_df = sample_df[sample_df["Sample Type"] == "Smp"]
            sample_df["Sample Name"] = sample_df["Filename"].apply(lambda x: "_".join(x.split("_")[2:-2]))
            sample_df = sample_df[["Filename", "Sample Name", "Adjusted Concentration"]]
            sample_df["Compound"] = compound
            concentration_data.append(sample_df)

    if concentration_data:
        df_concentrations = pd.concat(concentration_data, ignore_index=True)

        # Pivot using Filename
        df_concentrations_wide = df_concentrations.pivot(index="Filename", columns="Compound", values="Adjusted Concentration").reset_index()

        # Replace Filename with Sample Name after pivoting
        sample_name_map = df_concentrations.set_index("Filename")["Sample Name"].to_dict()
        df_concentrations_wide["Filename"] = df_concentrations_wide["Filename"].map(sample_name_map)
    else:
        df_concentrations_wide = pd.DataFrame()

    # ------------------ QC Sheet Data ------------------
    duplication_data = []
    spike_data = []
    for compound, data in filtered_compounds.items():
        if "Duplication Precision" in data and isinstance(data["Duplication Precision"], pd.DataFrame) and not data["Duplication Precision"].empty:
            dup_df = data["Duplication Precision"].copy()
            dup_df["Compound"] = compound
            dup_df["Precision (%)"] = dup_df["Precision"] * 100
            duplication_data.append(dup_df)

        if "Spike Precision" in data and isinstance(data["Spike Precision"], pd.DataFrame) and not data["Spike Precision"].empty:
            spike_df = data["Spike Precision"].copy()
            spike_df["Compound"] = compound
            spike_df["Precision (%)"] = spike_df["Precision"] * 100
            spike_data.append(spike_df)

    # Combine and flatten headers
    df_duplication = pd.concat(duplication_data, ignore_index=True) if duplication_data else pd.DataFrame()
    df_spike = pd.concat(spike_data, ignore_index=True) if spike_data else pd.DataFrame()

    # ------------------ Write to Excel ------------------
    with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
        workbook = writer.book
        header_format = workbook.add_format({'bold': True, 'font_size': 12, 'bg_color': '#C1E1FF', 'border': 1})
        text_format = workbook.add_format({'border': 1})

        # LOQ Sheet
        worksheet_loq = workbook.add_worksheet(loq_sheet_name)
        writer.sheets[loq_sheet_name] = worksheet_loq
        worksheet_loq.write(0, 0, "Compound", header_format)
        worksheet_loq.write(0, 1, "LOQ", header_format)
        for idx, row in df_loq.iterrows():
            worksheet_loq.write(idx + 1, 0, row["Compound"], text_format)
            if isinstance(row["LOQ"], (int, float)) and pd.notna(row["LOQ"]):  # Write numeric LOQ values
                worksheet_loq.write_number(idx + 1, 1, row["LOQ"], text_format)
            else:  # Leave blank for non-numeric or missing LOQ values
                worksheet_loq.write(idx + 1, 1, "", text_format)

        # Concentrations Sheet
        worksheet_concentrations = workbook.add_worksheet(concentration_sheet_name)
        writer.sheets[concentration_sheet_name] = worksheet_concentrations
        if not df_concentrations_wide.empty:
            for col_idx, col_name in enumerate(df_concentrations_wide.columns):
                worksheet_concentrations.write(0, col_idx, col_name, header_format)
            for row_idx, row in df_concentrations_wide.iterrows():
                for col_idx, value in enumerate(row):
                    if isinstance(value, (int, float)) and pd.notna(value):  # Write numeric values
                        worksheet_concentrations.write_number(row_idx + 1, col_idx, value, text_format)
                    else:  # Handle non-numeric values
                        worksheet_concentrations.write(row_idx + 1, col_idx, "<LOQ", text_format)

        # QC Sheet
        worksheet_qc = workbook.add_worksheet(qc_sheet_name)
        writer.sheets[qc_sheet_name] = worksheet_qc

        # Write the header for the compound columns
        worksheet_qc.write(0, 0, "Sample Name", header_format)
        for col_idx, compound in enumerate(compound_order, start=1):
            worksheet_qc.write(0, col_idx, compound, header_format)

        # Combine and align Duplication Precision and Spike Precision tables
        df_duplication_pivot = (
            df_duplication.pivot(index="Sample Name", columns="Compound", values="Precision (%)")
            if not df_duplication.empty
            else pd.DataFrame()
        )
        df_spike_pivot = (
            df_spike.pivot(index="Sample Name", columns="Compound", values="Precision (%)")
            if not df_spike.empty
            else pd.DataFrame()
        )

        # Get the union of all sample names
        all_sample_names = sorted(set(df_duplication_pivot.index).union(df_spike_pivot.index))

        # Write Duplication Precision
        worksheet_qc.write(1, 0, "Duplication Precision", header_format)
        for row_idx, sample_name in enumerate(all_sample_names, start=2):
            worksheet_qc.write(row_idx, 0, sample_name, text_format)
            for col_idx, compound in enumerate(compound_order, start=1):
                value = (
                    df_duplication_pivot.at[sample_name, compound]
                    if sample_name in df_duplication_pivot.index and compound in df_duplication_pivot.columns
                    else None
                )
                if isinstance(value, (int, float)) and pd.notna(value):
                    worksheet_qc.write_number(row_idx, col_idx, value, text_format)
                else:
                    worksheet_qc.write(row_idx, col_idx, "Can't calculate precision", text_format)

        # Leave a blank row, then write Spike Precision
        spike_start_row = len(all_sample_names) + 3
        worksheet_qc.write(spike_start_row, 0, "Spike Precision", header_format)
        for row_idx, sample_name in enumerate(all_sample_names, start=spike_start_row + 1):
            worksheet_qc.write(row_idx, 0, sample_name, text_format)
            for col_idx, compound in enumerate(compound_order, start=1):
                value = (
                    df_spike_pivot.at[sample_name, compound]
                    if sample_name in df_spike_pivot.index and compound in df_spike_pivot.columns
                    else None
                )
                if isinstance(value, (int, float)) and pd.notna(value):
                    worksheet_qc.write_number(row_idx, col_idx, value, text_format)
                else:
                    worksheet_qc.write(row_idx, col_idx, "Can't calculate precision", text_format)




    print(f"Excel file saved to '{output_file}' with three sheets: {loq_sheet_name}, {concentration_sheet_name}, and {qc_sheet_name}.")

