# Fake Reviews Detector

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)

## Project Overview

This project is a fake review detector built using a pre-trained Transformer model. The goal is to classify hotel reviews as either fake or genuine. The model is fine-tuned on a specific dataset to achieve this classification task, and the process is detailed in the `Fake_Reviews_Detector.ipynb` Jupyter Notebook.

## Features

- **Transformer-based Model**: Utilizes a powerful pre-trained Transformer model for state-of-the-art text classification.
- **Fine-tuning**: The model is fine-tuned on a custom dataset of hotel reviews.
- **Comprehensive Workflow**: The notebook covers the entire machine learning pipeline, from data loading and preprocessing to model training, evaluation, and prediction.

## Technologies Used

- **Python**: The core language for the project.
- **PyTorch**: The deep learning framework.
- **Hugging Face Transformers**: Provides the pre-trained model and tokenizer.
- **Pandas**: Used for data manipulation.
- **scikit-learn**: For dataset splitting and performance metric calculations.

## Getting Started

### Prerequisites

To run this project, you need to have Python installed along with the following libraries:

```bash
pip install torch transformers pandas scikit-learn
```

### Running the Notebook

- **Clone the repository (if applicable)**.
- **Download the dataset:** Place your dataset of hotel reviews in a data/ directory. The notebook assumes a specific format, which you may need to adjust.
- **Open the notebook:** Launch a Jupyter Notebook or JupyterLab environment.
- **Execute cells:** Run the cells in Fake_Reviews_Detector.ipynb sequentially to train and evaluate the model.

## Project Structure
- Fake_Reviews_Detector.ipynb: The main notebook containing all the code for the fake review detection model.
- data/: (Not included) This directory is where you should place your dataset.
- model/: (Generated during runtime) This directory will store the saved trained model and its tokenizer.

