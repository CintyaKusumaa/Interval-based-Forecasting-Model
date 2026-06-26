# Interval-Based Hybrid Framework for Volatile Time Series Forecasting

## Overview
This repository houses the core algorithmic functions utilized for the modeling and comparative study of interval-based approaches in forecasting highly volatile time series data. Harnessing advanced hybrid methodologies, the codebase streamlines the integration of Variational Mode Decomposition (VMD), Autoregressive Conditional Interval (ACI), and Interval Long Short-Term Memory (iLSTM) networks. The modules are designed to demonstrate a strong command of cutting-edge predictive modeling tailored for complex financial environments.

## Repository Structure
The repository is systematically organized to separate the core computational logic from the visual evaluation of model performance:

* **`Models/`**
  This directory exclusively contains the core Python functions (`.py`) utilized throughout the modeling process:
  * `vmd-aci-ilstm.py`: Functions defining the comprehensive hybrid framework combining VMD, ACI, and iLSTM.
  * `vmd-aci.py`: Functions integrating VMD with ACI models.
  * `vmd-ilstm.py`: Functions coupling VMD with the iLSTM network structure.
  * `ilstm.py`: Core computational functions for the Interval Long Short-Term Memory architecture.
  * `lstm.py`: Standard Long Short-Term Memory functions utilized for baseline comparative analysis.
  * `aci.py`: Functions implementing the Autoregressive Conditional Interval approach.
  * `garch.py`: Standard Generalized Autoregressive Conditional Heteroskedasticity (GARCH) functions for volatility modeling benchmarks.

* **`Plots/`**
  This directory contains the visual outputs and comparative forecasting plots generated during the experimental phase. The results are categorized by the specific volatile dataset analyzed:
  * `Gold Futures/`: Forecasting visualizations and error comparisons for Gold market data.
  * `JKSE/`: Forecasting visualizations and error comparisons for the Jakarta Composite Index.
  * `WTI/`: Forecasting visualizations and error comparisons for West Texas Intermediate crude oil.

## Academic References
The algorithms and mathematical logic formulated in these modules are strictly anchored in the theoretical frameworks established by the following literature:

1.  **Han, A., Hong, Y., & Wang, S. (2012).** *Autoregressive Conditional Models for Interval-Valued Time Series Data.* Working Paper, Department of Economics, Cornell University.
2.  **He, K., Sun, Y., dan Wang, S. (2021).** *Forecasting crude oil price intervals and return volatility via ACI models*. Econometric Reviews, 40(6), 584–606. https://doi.org/10.1080/07474938.2021.1889202
3.  **Zheng, L., Sun, Y., & Wang, S. (2024).** *A novel interval-based hybrid framework for crude oil price forecasting and trading.* Energy Economics, 130, 107266. https://doi.org/10.1016/j.eneco.2023.107266

## Contact & Implementation
The scripts provided are modular components utilized strictly during the experimental and modeling phases. For complete syntax guidelines, data pipeline instructions, or inquiries regarding the practical deployment of these functions, please reach out to the author directly.

**Email:** cintyakusumawardhani@gmail.com
