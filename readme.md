
# Informal Leases in the Upper Colorado River Basin

**Evaluating the potential of informal lease transfers to improve the re-allocation efficiency of prior appropriation-based institutions**

Harrison Zeff<sup>1\*</sup>, Antonia Hadjimichael<sup>2</sup>, Patrick Reed<sup>3</sup>, and Gregory W. Characklis<sup>1</sup>,

<sup>1 </sup> Center on Financial Risk in Environmental Systems, University of North Carolina at Chapel Hill
<sup>2 </sup> Department of Geosciences, Pennsylvania State University
<sup>3 </sup> Department of Civil and Environmental Engineering, Cornell University


\* corresponding author:  hbz5000@gmail.com

## Abstract
The ability to reallocate water to higher-value uses during drought is an increasingly important ‘soft-path’ tool for managing water resources in an uncertain future. In most of the Western  United States, state-level water market institutions that enable reallocation also impose substantial transaction costs on market participants related to regulatory approval and litigation. These transaction costs can be prohibitive for many participants in terms of both costs and lengthy approval periods, limiting transfers and reducing allocation efficiency, particularly during drought crises periods. This manuscript describes a mechanism to reduce transaction costs by adapting an existing form of informal leases to facilitate quicker and less expensive transfers among market participants. Instead of navigating the formal approval process to lease a water right, informal leases are financial contracts for conservation that enable more junior holders of existing rights to divert water during drought, thereby allowing the formal transfer approval process to be bypassed. The informal leasing approach is tested in the Upper Colorado River Basin (UCRB), where drought and institutional barriers to transfers lead to frequent shortages for urban rights holders along Colorado’s Front Range.  Informal leases are facilitated via option contracts that include adaptive triggers and that define volumes of additional, compensatory, releases designed to mitigate impacts to instream flows and third parties. Results suggest that more rapid reallocation of water via informal leases could have resulted in up to $222 million in additional benefits for urban rights holders during the historical period 1950 – 2013.

## Data reference
Source for StateMod_Model_15.exe and associated model runfiles (cm2015X.xxx) is here: https://github.com/antonia-had/cm2015_StateMod, which is in turn from the Upper Colorado Basin dataset here:  https://cdss.colorado.gov/modeling-data/surface-water-statemod
Shapefile data for model structures is taken from here: https://cdss.colorado.gov/gis-data/division-5-colorado

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| StateMod | Model_15 | executable in Github |  https://cdss.colorado.gov/modeling-data/surface-water-statemod |

## Reproduce my experiment
Fill in detailed info here or link to other documentation that is a thorough walkthrough of how to use what is in this repository to reproduce your experiment.


1. Install python library dependencies, cinluding
   a. pandas
   b. datetime
   c. numpy
   d. geopandas
   e. scipy
   f. matplotlib
   g. seaborn
   
2. Download this github repo and set as working directory
3. Run the following scripts to re-create this experiment:

| Script Name | Description | How to Run |
| --- | --- | --- |
| `ucrb_main.py` | Script to execture StateMod executable and evaluate informal leases
| `ucrb_postprocess.py` | Script to read StateMod output files generated from ucrb_main.py and generate analysis of informal leasing scenarios
| `ucrb_make_plots.py` | Script to plot analysis data from ucrb_postprocess.py
