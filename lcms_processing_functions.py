import pandas as pd
import numpy as np
from typing import Dict, Union
from sklearn.linear_model import LinearRegression

#### File Formatting  and Calibration ####

# Process the Calibration samples
def process_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Take a compound-specific dataframe and get the calibration curve and LOQ"""
    cal_std_df = df[df['Sample Type'] == 'Cal Std']
    cal_std_df = cal_std_df[(cal_std_df['Excluded'] != True) & (cal_std_df['Peak Label'] == 'T1')]
    
    cal_std_df = calculate_response_ratio(cal_std_df)
    LOQ = cal_std_df['Theoretical Amt'].min()

    return cal_std_df, LOQ

# Process all samples which are not for Calibration
def process_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Take a compound-specific dataframe and get the calibration curve and LOQ"""
    samples_df = df[df['Sample Type'] == 'Unknown']
    samples_df = samples_df[samples_df['Peak Label'] == 'T1']
    samples_df = calculate_response_ratio(samples_df)
    return samples_df

# Calculate the Response Ratio from the Area and ISTD Response columns
def calculate_response_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """Add Response Ratio column."""
    df.loc[:, 'Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df.loc[:, 'ISTD Response'] = pd.to_numeric(df['ISTD Response'], errors='coerce')
    df.loc[:, 'Response Ratio'] = df['Area'] / df['ISTD Response']
    return df

# Calculate the calibration curve
def get_calibration_curve(cal_std_df: pd.DataFrame, LOQ: float) -> Dict[str, float]:
    valid_data = cal_std_df[cal_std_df['Theoretical Amt'].notna() & cal_std_df['Response Ratio'].notna()]

    if len(valid_data) > 1:
        model = LinearRegression()
        model.fit(valid_data[['Theoretical Amt']], valid_data['Response Ratio'])
        slope = model.coef_[0]
        intercept = model.intercept_

        return {
            'slope': slope,
            'intercept': intercept
        }
    else:
        return None

# Use the calibration curve to calculate the concentrations
def calculate_concentrations(df: pd.DataFrame, LOQ: float, cal_curve: Union[Dict[str, float], None]) -> pd.DataFrame:
    """
    Calculates the concentration of samples based on a given calibration curve.
    If the calibration curve data is unavailable, it sets 'Concentration' to '<LOQ'.

    Parameters:
    df (pd.DataFrame): DataFrame containing at least the 'Response Ratio' column.
    LOQ (float): The limit of quantification; concentrations below this are set to NaN.
    cal_curve (Dict[str, float] or None): Dictionary containing the calibration curve 'slope' and 'intercept',
                                          or None if calibration data is insufficient.

    Returns:
    pd.DataFrame: DataFrame with an added 'Concentration' column with calculated concentrations or '<LOQ'.
    """
    if cal_curve is None:
        # Set the Concentration column to '<LOQ' if no calibration curve data
        df['Concentration'] = '<LOQ'
        return df
    
    # Proceed with calculation if cal_curve has valid data
    slope, intercept = cal_curve['slope'], cal_curve['intercept']
    df.loc[:, 'Concentration'] = ((df['Response Ratio']).astype(float) - intercept) / slope
    
    # Set concentrations below the LOQ to NaN
    if LOQ is not None:
        df.loc[df['Concentration'] < LOQ, 'Concentration'] = np.nan
    
    return df

# Get the sample type from the metadate after the first _ in the filename
def sample_type(df):
    """
    Categorizes each sample based on the text between the first and second underscores
    in the 'Filename' column and renames the 'Sample Type' column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'Filename' column.

    Returns:
    - pd.DataFrame: Updated DataFrame with a new 'Sample Type' column.
    """
    def get_sample_type(filename):
        if isinstance(filename, str):
            parts = filename.split('_')
            if len(parts) > 1:
                return parts[1]  # Sample type is the text between the first and second underscore
        return ""
    
    df['Sample Type'] = df['Filename'].apply(get_sample_type)
    return df

#### Blank Subtraction, Replicate Stats ####

# Blank subtraction function 
def subtraction(samples_df):
    """
    Adjusts the concentration of samples based on 'Sample Type' by:
    - Subtracting the concentration of the corresponding 'EB' control with the same 'EB_##' suffix.
    - For 'Spike' samples, additionally subtracting the corresponding 'Smp' concentration.
    - If the concentration is '<LOQ', sets the adjusted concentration to '<LOQ'.
    - If the adjusted concentration is negative, sets it to zero.

    Parameters:
    samples_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Concentration' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with adjusted concentrations for tagged samples.
    """

    eb_concentrations = {}
    smp_concentrations = {}
    adjusted_concentrations = []

    for idx, row in samples_df.iterrows():
        sample_type = row['Sample Type']
        filename = row['Filename']
        concentration = row['Concentration']
        
        # Only extract suffix for 'EB' samples
        if sample_type == 'EB' and '_EB_' in filename:
            eb_suffix = filename.split('_EB_')[-1]
            eb_concentrations[eb_suffix] = concentration
        elif sample_type == 'Smp':
            # Extract identifier for Smp samples
            identifier = '_'.join(filename.split('_')[2:-2])
            smp_concentrations[identifier] = concentration

    # Calculate adjusted concentrations
    for idx, row in samples_df.iterrows():
        sample_type = row['Sample Type']
        filename = row['Filename']
        concentration = row['Concentration']
        
        if concentration == '<LOQ':
            adjusted_concentrations.append('<LOQ')
            continue  # Skip to the next row
        
        # Start with the original concentration and then adjust for EB concentration
        adjusted_concentration = concentration
        eb_suffix = filename.split('_EB_')[-1] if '_EB_' in filename else None
        if sample_type in {'Smp', 'Dup', 'Spike'} and eb_suffix in eb_concentrations:
            eb_concentration = eb_concentrations[eb_suffix]
            if eb_concentration == '<LOQ':
                adjusted_concentration = '<LOQ'
            else:
                adjusted_concentration -= eb_concentration
                adjusted_concentration = max(adjusted_concentration, 0)  # Set to zero if negative
            
        # Additional subtraction for 'Spike' samples based on Smp identifier
        if sample_type == 'Spike':
            identifier = '_'.join(filename.split('_')[2:-2])
            if identifier in smp_concentrations:
                smp_concentration = smp_concentrations[identifier]
                if smp_concentration != '<LOQ' and adjusted_concentration != '<LOQ':
                    adjusted_concentration -= smp_concentration
                    adjusted_concentration = max(adjusted_concentration, 0)  # Set to zero if negative

        adjusted_concentrations.append(adjusted_concentration)

    samples_df['Adjusted Concentration'] = adjusted_concentrations

    return samples_df

# Calculate statistics on replicate samples
def replicate_stats(samples_df):
    """
    For each 'Smp' sample, identifies all replicates with the same sample name based on naming convention.
    Calculates the average and standard deviation of the 'Adjusted Concentration' for each group of replicates.

    Parameters:
    samples_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Adjusted Concentration' columns.

    Returns:
    pd.DataFrame: A new DataFrame with 'Sample Name', 'Average Concentration', and 'Standard Deviation' columns.
    """
    # Extract the core sample name based on the naming convention
    def extract_sample_name(row):
        if row['Sample Type'] == 'Smp':
            parts = row['Filename'].split('_')
            return '_'.join(parts[2:-3])
        return None
    
    samples_df['Sample Name'] = samples_df.apply(extract_sample_name, axis=1)
    samples_df['Adjusted Concentration'] = pd.to_numeric(samples_df['Adjusted Concentration'], errors='coerce')
    smp_df = samples_df[samples_df['Sample Type'] == 'Smp'].dropna(subset=['Sample Name'])

    # Group by 'Sample Name' and calculate the average and standard deviation of 'Adjusted Concentration'
    replicate_stats = smp_df.groupby('Sample Name')['Adjusted Concentration'].agg(
        Average_Concentration='mean', Standard_Deviation='std', Replicate_Count='count'
    ).reset_index()

    return replicate_stats

#### QC ####

# For duplicates:
def duplication_comparison(samples_df):
    """
    Compares the adjusted concentrations of samples with 'Smp' and 'Dup' Sample Types.
    Calculates the relative difference for pairs with matching 'Sample Name' identifiers.

    Parameters:
    samples_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Adjusted Concentration' columns.

    Returns:
    pd.DataFrame: DataFrame with the relative differences between 'Smp' and 'Dup' pairs.
    """
    precision_results = []

    # Extract 'Sample Name' for identifying pairs based on filename structure
    samples_df['Sample Name'] = samples_df.apply(
        lambda row: '_'.join(row['Filename'].split('_')[2:-2]) if row['Sample Type'] in ['Smp', 'Dup'] else "", axis=1
    )
    
    smp_df = samples_df[samples_df['Sample Type'] == 'Smp']
    dup_df = samples_df[samples_df['Sample Type'] == 'Dup']
    smp_dict = smp_df.set_index('Sample Name')['Adjusted Concentration'].to_dict()
    
    # Compare each 'Dup' sample with the matching 'Smp' sample using 'Sample Name' as the key
    for idx, row in dup_df.iterrows():
        sample_name = row['Sample Name']
        dup_concentration = row['Adjusted Concentration']
        
        if sample_name in smp_dict:
            smp_concentration = smp_dict[sample_name]
            
            # Calculate the precision in adjusted concentration
            if isinstance(smp_concentration, (int, float)) and isinstance(dup_concentration, (int, float)):
                precision = dup_concentration / smp_concentration if smp_concentration != 0 else None
                
                precision_results.append({
                    'Sample Name': sample_name,
                    'Smp Concentration': smp_concentration,
                    'Dup Concentration': dup_concentration,
                    'Precision': precision
                })
                
    dup_precision_df = pd.DataFrame(precision_results)
    return dup_precision_df

# For spikes:
def spike_comparison(samples_df):
    """
    Compares the adjusted concentrations of samples with 'Spike' and 'Ctrl' Sample Types.
    Calculates the relative difference for pairs with matching 'Control Name' identifiers embedded within filenames.

    Parameters:
    samples_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Adjusted Concentration' columns.

    Returns:
    pd.DataFrame: A new DataFrame with the relative differences between 'Spike' and 'Ctrl' pairs.
    """
    precision_results = []

    # Extract 'Control Name' based on Spike and Ctrl naming conventions
    def extract_sample_name(row):
        if row['Sample Type'] == 'Spike':
            return '_'.join(row['Filename'].split('_')[-4:-2])
        elif row['Sample Type'] == 'Ctrl':
            return '_'.join(row['Filename'].split('_')[1:]) 
        return ""
    
    samples_df['Control Name'] = samples_df.apply(extract_sample_name, axis=1)
    spike_df = samples_df[samples_df['Sample Type'] == 'Spike']
    ctrl_df = samples_df[samples_df['Sample Type'] == 'Ctrl']

    ctrl_dict = ctrl_df.set_index('Control Name')['Adjusted Concentration'].to_dict()
    
    # Compare each 'Spike' sample with the matching 'Ctrl' sample using 'Control Name' as the key
    for idx, row in spike_df.iterrows():
        sample_name = row['Control Name']
        spike_concentration = row['Adjusted Concentration']
        
        if sample_name in ctrl_dict:
            ctrl_concentration = ctrl_dict[sample_name]
            
            # Calculate the precision in adjusted concentration
            if isinstance(spike_concentration, (int, float)) and isinstance(ctrl_concentration, (int, float)):
                precision = spike_concentration / ctrl_concentration if ctrl_concentration != 0 else None
                
                # Store the results
                precision_results.append({
                    'Control Name': sample_name,
                    'Ctrl Concentration': ctrl_concentration,
                    'Spike Concentration': spike_concentration,
                    'Precision': precision
                })
        else:
            # Debug: Show if a matching Ctrl sample is not found
            print(f"No matching Ctrl sample found for Spike '{sample_name}'")

    spike_precision_df = pd.DataFrame(precision_results)
    return spike_precision_df

# For QC samples:
def qc_comparison(samples_df, cal_std_df):
    """
    Compares the concentrations of QC samples with corresponding calibration standards.
    Calculates the relative difference for pairs with matching calibration identifiers.

    Parameters:
    samples_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Adjusted Concentration' columns for QC samples.
    cal_std_df (pd.DataFrame): DataFrame containing 'Filename', 'Sample Type', and 'Concentration' columns for Calibration standards.

    Returns:
    pd.DataFrame: A new DataFrame with the differences between matching 'QC' and 'Calibration' pairs.
    """

    precision_results = []

    def extract_calibration_identifier(row):
        if row['Sample Type'] == 'QC':
            return '_'.join(row['Filename'].split('_')[2:4])  # Identifier for QC
        elif row['Sample Type'] == 'Cal':
            return '_'.join(row['Filename'].split('_')[1:3])  # Identifier for Calibration
        return None
    
    samples_df['Calibration Identifier'] = samples_df.apply(extract_calibration_identifier, axis=1)
    cal_std_df['Calibration Identifier'] = cal_std_df.apply(extract_calibration_identifier, axis=1)
    qc_df = samples_df[samples_df['Sample Type'] == 'QC']
    cal_dict = cal_std_df.set_index('Calibration Identifier')['Concentration'].to_dict()

    # Compare each QC sample with the matching Calibration standard using the calibration identifier
    for idx, row in qc_df.iterrows():
        calibration_id = row['Calibration Identifier']
        qc_concentration = row['Adjusted Concentration']
        
        if calibration_id in cal_dict:
            cal_concentration = cal_dict[calibration_id]
            
            # Calculate the precision
            precision = cal_concentration / qc_concentration if qc_concentration != 0 else None
            
            precision_results.append({
                'Calibration Identifier': calibration_id,
                'Calibration Concentration': cal_concentration,
                'QC Concentration': qc_concentration,
                'Precision': precision
            })

#### OUTPUT ####
def to_excel(compound_dict, output_file, sheet_name="All_Compounds"):
    """
    Saves Duplication Precision, Spike Precision, QC Precision, and Replicate Statistics
    for all compounds in a single sheet in an Excel file, with custom formatting.

    Parameters:
    - compound_dict (dict): Dictionary containing processed data for each compound.
    - output_file (str): Path to the output Excel file.
    - sheet_name (str): Name of the sheet where data will be saved.
    """
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = worksheet

        # Define formatting for compound names, table headers, and borders
        compound_name_format = workbook.add_format({'bold': True, 'font_size': 14})  # Larger, bold compound name
        header_formats = {
            "Duplication Precision": workbook.add_format({'bg_color': '#FFDDC1', 'bold': True}),
            "Spike Precision": workbook.add_format({'bg_color': '#C1E1FF', 'bold': True}),
            "QC Precision": workbook.add_format({'bg_color': '#D9FFC1', 'bold': True}),
            "Replicate Statistics": workbook.add_format({'bg_color': '#FFFAC1', 'bold': True}),
        }
        
        start_row = 0  # Track the starting row across all compounds in the same sheet

        # Iterate over each compound in the dictionary
        for compound_name, data in compound_dict.items():
            col_offset = 1 
            
            worksheet.write(start_row, 0, f"Compound: {compound_name}", compound_name_format)
            start_row += 1 
            
            # Function to write a table with bold headers and an outer border
            def write_table_with_format(df, header, start_row, start_col):
                df = df.replace([np.nan, np.inf, -np.inf], "N/A")
                worksheet.write(start_row, start_col, header, header_formats[header])
                df.to_excel(writer, sheet_name=sheet_name, startrow=start_row+1, startcol=start_col, index=False)
                end_row = start_row + len(df) + 1 
                end_col = start_col + len(df.columns) - 1
                border_format = workbook.add_format({'border': 1})
                worksheet.conditional_format(start_row, start_col, end_row, end_col, {'type': 'no_errors', 'format': border_format})

            # Write each table with specific header format and outer borders
            tables = {
                "Duplication Precision": data.get('Duplication Precision'),
                "Spike Precision": data.get('Spike Precision'),
                "QC Precision": data.get('QC Precision'),
                "Replicate Statistics": data.get('Replicate Statistics')
            }
            for table_name, df in tables.items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    write_table_with_format(df, table_name, start_row, col_offset)
                    col_offset += len(df.columns) + 2 

            start_row += max(
                len(tables["Duplication Precision"]) if tables["Duplication Precision"] is not None else 0,
                len(tables["Spike Precision"]) if tables["Spike Precision"] is not None else 0,
                len(tables["QC Precision"]) if tables["QC Precision"] is not None else 0,
                len(tables["Replicate Statistics"]) if tables["Replicate Statistics"] is not None else 0,
                1
            ) + 4

    print(f"Data for all compounds has been saved to '{output_file}' in the sheet '{sheet_name}'")