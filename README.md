# Optimization with `PyMoosh` and `Nevergrad`

Tutorial python notebooks, accompanying the paper *An illustrated tutorial on global optimization in nanophotonics* ([link](https://notyetthere)).

We demonstrate how to do global optimization of the geometry of multi-layered photonic structures. Multi-layer simulations are done with [`PyMoosh`](https://github.com/AnMoreau/PyMoosh), as optimization toolkit we use [`Nevergrad`](https://facebookresearch.github.io/nevergrad/).

## Goal of the example problems

**The tutorials have two main goals:**

  1. providing a simple introduction and a starting point for global optimization of multi-layer structures.
  
  2. demonstrating advanced benchmarking in order to find the best optimization algorithm for a specific problem and to assess the quality of the found solution.
  

**Three specific examples are treated:**

  1. optimization of a Bragg mirror
  
  2. Solving of an ellipsometry inverse problem
  
  3. design of a sophisticated antireflection coating to optimize solar absorption in a photovoltaic solar cell


## List of all notebooks

  - `01_getting_started_optimization_with_pymoosh.ipynb`: Very simple tutorial how to use `PyMoosh`'s internal DE optimizer ([link to google colab version](https://colab.research.google.com/drive/11il22JcUqIJbT6yCA7kbwuwHYAGgDVKD))
  
  - `02_getting_started_pymoosh_with_nevergrad.ipynb`: Very simple tutorial how to use `Nevergrad` optimizers with `PyMoosh` ([link to google colab version](https://colab.research.google.com/drive/1kWw10Gem4EmFot1YXPyixbJ1f8D5HxWO))
  
  - `03_algo_benchmark_pymoosh_with_nevergrad.ipynb`: Tutorial how to benchmark several algorithms, demonstrated on a small Bragg mirror problem ([link to google colab version](https://colab.research.google.com/drive/1VamY1EnzlbmfTmUWsP6fgbGDF5PyBAMn))
  
  - `04_ellipsometry_simple.ipynb`: Tutorial setting up the ellipsometry problem ([link to google colab version](https://colab.research.google.com/drive/1B3htjF8DkbxpKIawtJZapQboco1ECaCT))
  
  - `05_ellipsometry_algo_benchmark.ipynb`: Algorithm benchmark on the ellipsometry problem ([link to google colab version](https://colab.research.google.com/drive/1CihO6Sm4BDYeeJpA7N9faXtZ7sx5vtrn))
  
  - `06_photovoltaics_simple.ipynb`:  Tutorial setting up the photovoltaics problem ([link to google colab version](https://colab.research.google.com/drive/1qzBtezWNgfH2mFRuljXmGiBndYvaCAik))
  
  - `07_photovoltaics_algo_benchmark.ipynb`: Algorithm benchmark on the photovoltaics problem ([link to google colab version](https://colab.research.google.com/drive/13Y4A4wWmjBp4OoLJFyx_c4NIrSHLCFon))
    
  - `09_paper_results_reference_Pymoosh_and_Nevergrad.ipynb`: Reproduce all results of the paper ([link to google colab version](https://colab.research.google.com/drive/1Xk3gY1SK0xIivFRaCGMzsYrP_7RIbGmQ?usp=sharing))

An additional version with ``Centered'' as prefix is included, with initialization with 50\% matter instead of low proportion of matter.
## DOI

The DOI of this tutorial repository is: 
[![DOI]()]()

