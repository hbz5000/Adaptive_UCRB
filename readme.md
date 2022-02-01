_zenodo placeholder_

# zeff-etal_journal_tbd

**Evaluating the potential of informal lease transfers to improve the re-allocation efficiency of prior appropriation-based institutions**

Harrison Zeff<sup>1\*</sup>, Antonia Hadjimichael<sup>2</sup>, Patrick Reed<sup>3</sup>, and Gregory W. Characklis<sup>1</sup>,

<sup>1 </sup> Center on Financial Risk in Environmental Systems, University of North Carolina at Chapel Hill
<sup>2 </sup> Department of Geosciences, Pennsylvania State University
<sup>3 </sup> Department of Civil and Environmental Engineering, Cornell University


\* corresponding author:  zeff@live.unc.edu

## Abstract
The Colorado-Big Thompson (CBT) project enables Northern Water to divert up to 310k AF of water from the Colorado River Basin to support over 1 million residents and 600k irrigated acres in Northeastern Colorado, but annual deliveries are highly variable. Although permanent transfers are subject to costly and slow legal review, informal transfers can be used by trans-basin diverters to supplement dry-year supplies. We use StateMod, the State of Colorado’s Decision Support System, to evaluate the institutional capacity for adaptive demand coordination to increase dry year trans-basin diversions.

## Journal reference
Zeff. H.B, et al. (TBD). Evaluating the potential of informal lease transfers to improve the re-allocation efficiency of prior appropriation-based institutions (in preparation).

## Code reference: None of these are real links - we've yet to publish a final version of the code
Zenodo link:
Human, I.M. (2021, April 14). Project/repo:v0.1.0 (Version v0.1.0). Zenodo. http://doi.org/some-doi-number/zenodo.7777777

## Data reference

### Input data
Reference for each minted data source for your input data.  For example:

Human, I.M. (2021). My input dataset name [Data set]. DataHub. https://doi.org/some-doi-number

### Output data
Reference for each minted data source for your output data.  For example:

Human, I.M. (2021). My output dataset name [Data set]. DataHub. https://doi.org/some-doi-number

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateMod | Model_15 | executable in Github | link to DOI dataset |
| model 2 | version | link to code repository | link to DOI dataset |
| component 1 | version | link to code repository | link to DOI dataset |

## Reproduce my experiment
Fill in detailed info here or link to other documentation that is a thorough walkthrough of how to use what is in this repository to reproduce your experiment.


1. Install the software components required to conduct the experiement from [Contributing modeling software](#contributing-modeling-software)
2. Download and install the supporting input data required to conduct the experiement from [Input data](#input-data)
3. Run the following scripts in the `workflow` directory to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `step_one.py` | Script to run the first part of my experiment | `python3 step_one.py -f /path/to/inputdata/file_one.csv` |
| `step_two.py` | Script to run the last part of my experiment | `python3 step_two.py -o /path/to/my/outputdir` |

4. Download and unzip the output data from my experiment [Output data](#output-data)
5. Run the following scripts in the `workflow` directory to compare my outputs to those from the publication

| Script Name | Description | How to Run |
| --- | --- | --- |
| `compare.py` | Script to compare my outputs to the original | `python3 compare.py --orig /path/to/original/data.csv --new /path/to/new/data.csv` |

## Reproduce my figures
Use the scripts found in the `figures` directory to reproduce the figures used in this publication.

| Script Name | Description | How to Run |
| --- | --- | --- |
| `generate_figures.py` | Script to generate my figures | `python3 generate_figures.py -i /path/to/inputs -o /path/to/outuptdir` |
