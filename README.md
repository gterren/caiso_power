# Joint Day-Ahead Probabilistic Energy Forecast

A robust, scalable library for probabilistic day-ahead forecasting of electricity demand and renewable (solar & wind) generation in CAISO trading hubs and major utilities — leveraging publicly available weather data and machine learning methods. It also facilitates system-level joint uncertainty analysis and operating reserve planning.

### Key Features
* **Joint probabilistic forecasts:** Simultaneously model electricity demand, solar, and wind generation using joint probability distributions.
* **Sparse + Bayesian learning:** Combines four sparse learning techniques for feature selection and four Bayesian uncertainty quantification methods.
* **Model evaluation framework:** Rigorously assesses model combinations via proper scoring rules.
* **CAISO-specific zones:** Supports the three CAISO trading hubs (NP15, SP15, ZP26) and the three major utilities (PG&E, SCE, SDG&E), and produces a ~25% forecasting accuracy improvement.
* **Reserve allocation tool:** Uses forecast confidence intervals to optimize operating reserve levels beyond deterministic forecasts.

## Required packages

``conda install -c anaconda numpy pandas scipy scikit-learn`` 

``conda install scikit-learn-intelex``

``conda install -c conda-forge properscoring blosc mpi4py``

``pip install group-lasso``

## Code description


## Reference

The manuscript is currently undergoing revisions in Nature Communications. The draft is publicly available (https://www.researchsquare.com/article/rs-5891000/v1). We recommend using the following reference:

Terrén-Serrano, Guillermo, Ranjit Deshmukh, and Manel Martínez-Ramón. "Joint Probabilistic Day-Ahead Energy Forecast for Power System Operations." (2025).

