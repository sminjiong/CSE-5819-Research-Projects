# CSE-5819-Research-Projects
Spring 2026
CSE5819-Introduction to Machine Learning

## Overview
This project focuses on Sleep Apnea detection using St. Vincent's University Hospital/ University college Dublin Sleep Apnea Database.    
*Database URL: https://physionet.org/content/ucddb/1.0.0/*

The model was inspired by *"MAESTRO : Adaptive Sparse Attention and Robust Learning for Multimodal Dynamic Time Series"*, and the framework was adapted and modified based on the propsed architecture  
*Research Paper URL: https://arxiv.org/abs/2509.25278*  
*MAESTRO code: https://github.com/payalmohapatra/MAESTRO*

---

# Project Structure

```bash
CSE-5819-Research-Projects/
│
├── data_utils/
│   └── sleepapnea_dataset.py
│       # Dataset loading and preprocessing utilities
│
├── main/
│   ├── CSE5819_Final.py
│   │   # Final experiment pipeline, evaluation
│   └── main.py
│       # Main execution script, training and validation
│
├── models/
│   ├── our_models.py
│   │   # Model architectures 
│   └── train_utils.py
│       # Training, validation, and test functions
│
├── utils/
│   ├── dataset_cfg.py
│   │   # Dataset configuration settings
│   └── helper_function.py
│       # General helper functions
│
└── README.md
```

# Running the Project

Run the final pipeline:

```bash
python main/main.py
```

main.py saves the process and results in .pkl file
Once you run main.py you can evaluate the results in:

```bash
python main/CSE5819_Final.py
```
