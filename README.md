# Sustained Phonation Features

Code to replicate the features from the paper https://arxiv.org/abs/2006.05365 to appear at Interspeech 2020.


## Installing the environment

To run the experiment's notebook, the creation of a conda environment
is required. Make sure you have conda installed, and are `cd`'ed into 
the project's folder, and run

```shell script
conda env create -f environment.yml
conda activate pho_features
```

You should be then good to go

## Extracting the features

Make sure you have installed and activated the aforementioned conda environment,
then launch a jupyter notebook, and reach the `Sustained_Phonation.ipynb` file
in the interface.

```shell script
jupyter notebook 
```

## References

    .. [1] Riad, R, Titeux, H, Lemoine, L., Montillot J. Hamet Bagnou, J. Cao, X., Dupoux, E & Bachoud-Lévi A.-C.
           *Vocal markers from sustained phonation in Huntington's Disease.*
           In: INTERSPEECH-2020

If you use the Modulation Power Spectrum features, please cite also:

    .. [2] Elie, J. E. and F. E. Theunissen 
           *Zebra finches identify individuals using vocal signatures unique to each call type.*
           In: Nature Communications. 2018

