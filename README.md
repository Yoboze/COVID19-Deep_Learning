### Deep Learning (MTSU) Project

**Step 1:** Click on this link https://github.com/Yoboze/eo3j-Fall-2021-DL-DEMO.git. It will take you to the GitHub repository of eo3j-Fall-2022-DL-DEMO, which is the repository for my deep learning project.


**Step 2:** Download the zip file from the root page of this repository. Then, you will have immediate access to all the files needed to run this project. The project aims to predict the time that a plateau will be reached and the cumulative number of individuals reported to be infected by COVID-19. Italy and Sweden are the countries used as a case studies. Therefore, the user can explore the data for this project using world_confirmed.csv. The CSV file is inside each folder on the eo3j-Fall-2021-DL-DEMO repository. Four mathematical models and two deep learning models are the models used for this project. The folder Italy contains the python scripts for the four mathematical models, namely constant.py, Italy_rational.py, Italy_birational.py, and Italy_timeseries.py, which are used for Italy. The same with the folder name Sweden used for Sweden. The folder Italy_RNN and Sweden_RNN contain the python scripts for the deep learning models, namely LSTM and GRU.


**Step 3:** The text files named readme with model names demonstrated how to run each python script and generate the text file results from the outputs. **To run the python script, use the JupyterLab terminal**. The JupyterLab notebook named All_Italy_models.ipynb runs the output for four mathematical models for Italy and Sweden_Models.ipynb for Sweden. Italy_RNN.ipynb and Sweden_RNN.ipnb run the output for the python script Italy_RNN and Sweden_RNN for deep learning models (LSTM and GRU) for Italy and Sweden.

**A text is included at the top of the notebook explaining what the block of code is doing**

To execute a cell, click on it to select it and **Shift+Enter.**

Plots are shown to describe each result.

The Error Metrics such as **RMSE, MAPE, and EV** are obtained for the mathematical and deep learning models.

The necessary python packages needed to be installed for this project are:

* Numpy
* Matplotlib
* Sklearn
* Tensorflow
* Pytorch
* pyDOE
* sys
* scipy



