### MergerMonger: Identify different types of mergers in SDSS imaging

Based on the imaging classification presented in Nevin+2019 and Nevin+2022.

There are three main steps:
## 1) Creating the classification from simulations of merging galaxies:
<img src="images_for_github/panel_merger_timeline.png">

The simulations are fully described in Nevin+2019.

```
code example

```

## 2) Measure predictor values from images (GalaxySmelter):
I also include some utilities for visualizing individual galaxies and their predictor values.
<img src="images_for_github/prob_panel_low.png" alt="probability panel" width="500">

## 3) Classify galaxies and obtain merger probability values:
<img src="images_for_github/p_merg_recipe.png">

I also include some utilities for interpreting these probability values using the CDF of the full population.
