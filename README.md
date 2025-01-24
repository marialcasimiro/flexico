# FLEXICO: Sustainable Machine Translation via Self-Adaptation

If you find our repository useful for your research, please consider citing our paper:

```
@inproceedings{casimiro2025flexico,
  title={FLEXICO: Sustainable Machine Translation via Self-Adaptation},
  author={Casimiro, Maria and Romano, Paolo and Souza, José and Khan, Amin M. and Garlan, David},
  booktitle={Proceedings of the 20th International Symposium on Software Engineering for Adaptive and Self-Managing Systems},
  year={2025}
}
```

## More results not featured in the paper

We created an appendix with more information about our experimental results which you can consult in zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14452761.svg)](https://doi.org/10.5281/zenodo.14452761)




## Data for reproducing the results of the paper

The FIDs and other data files required to run Flexico and reproduce the results of the paper can be downloaded from zenodo: 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14736503.svg)](https://doi.org/10.5281/zenodo.14736503)

Prior to re-producing the results from the paper, it is **necessary to specify the path to the data files in file `src/constants.py`**.



## Required software

In order to reproduce the experiments, the user needs **version 4.8.1 of the PRISM model checker** (as described in the paper).

The PRISM model checker can be downloaded from its official page 
[https://www.prismmodelchecker.org/download.php](https://www.prismmodelchecker.org/download.php)
and installed on Linux, MacOS and Windows.

**The path to the PRISM executable must be specified in `src/constants.py` with the `PRISM` macro**.

Flexico relies on python: `requirements.txt` lists the required packages and corresponding versions.



## Running Flexico
To run Flexico or any of the other baselines described in the paper, one of the following commands can be executed. 

More information can be obtained with `--help`.

**Flexico**

- scenario A:
```
python3.8 src/flexico/scenarioA/run_adaptiveMT_framework.py
```

- scenario B:
```
python3.8 src/flexico/scenarioB/run_adaptiveMT_framework.py
```

**Optimum baseline**

- scenario A:
```
python3.8 src/flexico/scenarioA/run_global_optimum_baseline.py 
```

- scenario B:
```
python3.8 src/flexico/scenarioB/run_global_optimum_baseline.py
```

**Other baselines**

- scenario A:
```
python3.8 src/flexico/scenarioA/run_adaptiveMT_framework.py -b periodic-2 reactive-85 exponential-2 random-50
```

- scenario B:
```
python3.8 src/flexico/scenarioB/run_adaptiveMT_framework.py -b periodic-2 reactive-85 exponential-2 random-50
```