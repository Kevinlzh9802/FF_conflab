# F-formation detection baseline for Conflab 

This folder contains the code for obtaining FF baseline results using GCFF [1] and GTCG [2].

## System Requirements
These methods use Matlab on a Windows and Linux 64bit machine, otherwise you can recompile the mex file in each of the method.

## Overview 
[data_process](https://github.com/steph-tan/FF_conflab/tree/main/data_process) contains the code to extract orientations from pose data, and process the features to be fed into the methods. See [readme.txt](https://github.com/steph-tan/FF_conflab/blob/main/data_process/readme.txt) for more details. 

GCFF (example_GCFF.m) and GTCG (runner.m) contain code to run the method once the data is prepared and processed. See [readme of GCFF](https://github.com/steph-tan/FF_conflab/blob/main/GCFF/README.md) and [readme of GTCG](https://github.com/steph-tan/FF_conflab/blob/main/GTCG/README.md) for more details.  

## Citations

### [1] Citing GCFF

@article{setti2015f,

title={F-formation detection: Individuating free-standing conversational groups in images},

author={Setti, Francesco and Russell, Chris and Bassetti, Chiara and Cristani, Marco},

journal={PloS One},

volume={10},

number={5},

pages={e0123783},

year={2015},

publisher={Public Library of Science}

}

### [2] Citing GTCG

@article{vascon2016detecting,

title={Detecting conversational groups in images and sequences: A robust game-theoretic approach},

author={Vascon, Sebastiano and Mequanint, Eyasu Z and Cristani, Marco and Hung, Hayley and Pelillo, Marcello and Murino, Vittorio},

journal={Computer Vision and Image Understanding},

volume={143},

pages={11--24},

year={2016},

publisher={Academic Press}

}

@inproceedings{vascon2014game,

title={A Game-Theoretic Probabilistic Approach for Detecting Conversational Groups},

author={Vascon, Sebastiano and Eyasu, Zemene and Cristani, Marco and Hung, Hayley and Pelillo, Marcello and Murino, Vittorio},

booktitle={Asian Conference in Computer Vision},

year={2014}

}


