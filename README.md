# Zezima: Gene Prediction with Transformer Model

## Overview
Zezima is a sophisticated gene prediction tool that leverages the power of Transformer models to analyze DNA sequences. <br>
It's designed to efficiently predict gene-related features within DNA sequences, aiding in the complex task of genomic analysis.

## Key Features
- **Transformer Model**: Utilizes an advanced machine learning approach for accurate gene prediction.
- **Custom DNA Sequence Handling**: Tailored to process specific DNA sequence formats.
- **Configurable Parameters**: Offers flexibility to adjust model parameters in `config.py`

## Requirements

Before beginning the setup, ensure your system meets the following requirements:
- **Operating System:** Linux
- **Python Version:** Python 3.10 or higher


### Setting Up Python Virtual Environment
A Python virtual environment is recommended for managing the project's dependencies.
Follow these steps to create and activate your virtual environment:

1. **Create a Virtual Environment:**
   ```bash
   python3.10 -m venv venv
   ```

2. **Activate the Virtual Environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Install Required Python Packages:**
   ```bash
   pip3 install -Ur requirements.txt
   ```

## Data Format
The input data should be structured as follows:
```
#HEADER#
#DATE: YYYY-MM-DD
#bp_vector_schema: ['A', 'C', 'G', 'T', 'PROMOTOR_MOTIF', 'ORF', 'exon', 'mRNA', 'miRNA', 'rRNA', 'CDS', 'POLY_ADENYL', 'gene']
#description of feature: 0 = no present, 1 = start, 2 = continuation/ongoing, 3 = end
####END####
[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, [1], [1, 2]]
```



## Usage
1. Prepare your DNA sequence data files according to the specified format and
place them in the designated input directory.
2. Configure the model and execution parameters in the `config.py` file.
3. Run the application:

    ```bash
    python3.10 run.py
   ```

## License
Zezima is open-source software licensed under the MIT License.
For more details, see the `LICENSE` file.
