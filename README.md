# Zezima: Gene Prediction with Transformer Model

## Overview
Zezima is a sophisticated gene prediction tool that leverages the power of Transformer models to analyze DNA sequences. <br>
It's designed to efficiently predict gene-related features within DNA sequences, aiding in the complex task of genomic analysis.

## Key Features
- **Transformer Model**: Utilizes an advanced machine learning approach for gene prediction.
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
The input data file should be structured as follows, containing a header section and subsequent data vectors:
```
#HEADER#
#DATE=YYYY-MM-DD
#pre_processing_version=[0, 1, 0]
#bp_vector_schema=['A', 'C', 'G', 'T', 'PROMOTOR_MOTIF', 'ORF', 'exon', 'mRNA', 'miRNA', 'rRNA', 'CDS', 'POLY_ADENYL', 'gene']
#description of feature:[0, 0, 0]=no_present, [1, 0, 0]=start, [0, 1, 0]=continuation/ongoing, [0, 0, 1]=end
#max_feature_overlap=1
[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

Each vector in the file represents information about a specific position on the DNA,
as defined by `bp_vector_schema`. The vector encodes the presence and status of
various genetic features at that position.

For more detailed information about the input data structure and the pre-processing steps,
please refer to the [pre-processing documentation](https://github.com/ProfH2SO4/lumbridge?tab=readme-ov-file#structure-of-model-file).


## Usage
1. Prepare your DNA sequence data files according to the specified format and
place them in the designated input directory. In a case that you don't have your own pipelines
you can use or modify [this](https://github.com/ProfH2SO4/lumbridge?tab=readme-ov-file) pre-processing stage to your needs
2. Configure the model and execution parameters in the `config.py` file.
3. Run the application:

    ```bash
    python3.10 run.py
   ```

## License
Zezima is open-source software licensed under the MIT License.
For more details, see the `LICENSE` file.
