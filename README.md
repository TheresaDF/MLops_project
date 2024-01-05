# VAE_cats

Generate cats using a VAE

## Project description
Our goal of this project is to brighten the world with more cat pictures, by using variational auto encoders (VAE) to generate new images of cats. As our goal of the project primarily relates to generation it motivates the use of VAE rather than ordinary auto encoders. 

To obtain this, we will seek inspiration from various frameworks such as pre-trained models from HuggingFace and high-level training frameworks.  

The dataset used is provided by the DTU course, 02502 Image Analysis, which contains $1706$ RGB images of cats. The data has already been preprocessed in this course, with the use of landmarks, to a standardized size of $(360\times360\times3)$, and to center the cat faces. We will likely preprocess the images further as we see fit, to adjust for upcoming challenges. The original dataset is from Kaggle and can be found [here](https://www.kaggle.com/datasets/crawford/cat-dataset). 

As previously mentioned, a variational auto-encoder model will be used, to generate cat images. During this project, we will experiment with different numbers of hidden layers and neurons in the layers to obtain various models. 


## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── VAE_cats  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
