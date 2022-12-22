MAGNET

The repository has the following structure:


    - baseline
                      - GCN-Align with anatomy
                                              - baseline_exec.ipynb
                                              - data_processing.ipynb
                                              - plot.ipynb

                      - MAGNET with dbp15k
                                              - baseline_exec.ipynb
                                              - data_processing.ipynb
                      - SBERT
                                              - baseline_exec.ipynb

     - data
                      - 1_raw
                                              - anatomy
                                              - common knowledge graphs
                                              - dbp15k
                      - 2_preprocessed
                                              - anatomy
                                              - common knowledge graphs
                                              - dbp15k
     - data preparation
                      - AlignmentFormat.py
                      - create_graph.py
                      - execution.ipynb
                      - parse_graph.py
     - evaluation
                      - epoch_analysis.ipynb
                      - error_analysis.ipynb
                      - interpretation.py
                      - interpretation_exec.ipynb
     - model execution
                      - model_exec_GAT.ipynb
                      - model_exec_GCN.ipynb
                      - model_exec_RGAT.ipynb
                      - postprocessing.ipynb
     - model functions
                      - helper_funcs.py
                      - losses.py
                      - model_definitions.py
     - results
                      - trained_models
                      - trainings
                      - translation.txt



The necessary libraries are installed in the respective notebooks.

- Baseline: The folder "baseline" contains all three baselines containing both data preparation (if applicable) and the execution of the respective model.
- Data (on stick): Contains both raw as well as preprocessed datasets
- Data Preparation: The folder "data preparation" contains both the python files for data preparation as well as the execution notebook.
- Evaluation: The folder "evaluation" contains all notebooks for different types of model result analysis.
Specifically, epoch_analysis.ipynb plots the loss & performance metrics of the important models. error_analysis.ipynb
is used to plot the frequency of the central nodes and the amount of model guesses. In interpretation_exec.ipynb, the evaluation
part of the thesis is conducted. In particular, confidence, match level analysis and data construction is looked at.
The interpretation.py contains additional functions to analyse results.
- Model Execution: This folder contains the execution files for the three main models with default settings. This distinction
is due to easier parallel execution of models. Also, it contains the postprocessing steps.
- Model Functions: Here, general models and functions to run them are contained. helper_funcs.py contains several helper functions.
losses.py contains all tested loss versions. model_definitions.py contains the model definitions.
- Results (on stick): This folder contains a selection of some trained models and training results. Also, a translation of the most
important model runs is contained.

