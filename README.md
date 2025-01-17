**lcms_processing.ipynb** is an _Interactive Python Notebook_ which takes in an Excel file from the LC-MS (after certification of the calibration points and peak selection) and can be used to obtain blank-subtracted concentrations, limit of quantifications, and quality control data.

**lcms_processing_functions.py** is a _Python Script_ which performs the actions of formatting the LC-MS Excel file, calculating a calibration curve for each compound, calculating concentrations of samples, subtracting the extraction blank concentration, then calculating replicate statistics and precisions for duplicate, spike, and QC samples.

**output_dataframes.py** is a _Python Script_ which formats the processed data (including LOQ, concentrations, and precisions) in a way which can be easily visualized at the end of lcms_processing.ipynb

**name_replacement.ipynb** is an _Interactive Python Notebook_ which can be used to easily reformat sample names to be compatible with the processing script.

**LC-MS Procedural Naming Guide.docx** is a _Word Document_ which provides instructions for how to designate each type of measurement (calibration, blank, control, sample, duplicate, spike, extraction blank, quality control, and wash) to be compatible with the processing script.

_Written by Ethan J. Sontarp and Colin P. Thackray_
