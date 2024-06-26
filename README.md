#  Quantifying Pressure in Cricket for building Expected Runs model

## Update: The recipe data project can be found here - [https://github.com/Rit-ctrl/Recipe-Data-Analysis](https://github.com/Rit-ctrl/Recipe-Data-Analysis).

Sorry for the inconvenience 

![Flowchart of proposed model](flowchart_page-0001.jpg)

## Agenda 
- Utilise LSTM model to build pressure index
- Combine this with current state variables to predict runs scored
- Use other methods from literature
- Build models without previous state information to ascertain if pressure is useful for predicting runs scored
- IPL play-by-play data obtained from [Cricsheet.org](https://cricsheet.org)
- Train on IPL seasons 2008-2021, test on 2022-2023 
- Compare Mean squared error for the models to evaluate


   <img width="569" alt="image" src="https://github.com/Rit-ctrl/Pressure-Quantification-for-Cricket/assets/53635318/6bef5314-4a7c-4497-8bac-6c560f0c4c89">


## Code structure

- [utils.py](utils.py) : Contains helper functions for reading the input files
   -  [process_df](https://github.com/Rit-ctrl/Pressure-Quantification-for-Cricket/blob/31e1778576ed66a8da6360c6e2d0ccbebb6f4676/utils.py#L10) : reads csv files and processes them
  -  [get_data](https://github.com/Rit-ctrl/Pressure-Quantification-for-Cricket/blob/31e1778576ed66a8da6360c6e2d0ccbebb6f4676/utils.py#L53C1-L53C4) : gets training and test data
  -  [xR_Model](https://github.com/Rit-ctrl/Pressure-Quantification-for-Cricket/blob/31e1778576ed66a8da6360c6e2d0ccbebb6f4676/utils.py#L139C1-L140C1) : Pytorch model definition for proposed model

- [00_Read.ipynb](00_Read.ipynb) : Notebook for exploring the data. Useful if you want to take a look at the data 

- [01_PI_baseline.ipynb](01_PI_baseline.ipynb) : Notebook for calculating MSE for baseline pressure index PI<sub>3</sub>

- [02_Results.ipynb](02_Results.ipynb) : Notebook for calculating MSE for proposed model, as well as Linear Regression and xGBoost Model

- [DLR.csv](DLR.csv) contains the Duckworth-Lewis Resource table used for calculating the PI<sub>3</sub>

