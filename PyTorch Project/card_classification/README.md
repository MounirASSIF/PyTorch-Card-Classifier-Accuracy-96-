# Card Classification Project

This project implements an image classification model to detect playing cards using PyTorch. The model is trained on a dataset of playing card images and can classify them into different categories.

[👉 Run on Kaggle](https://www.kaggle.com/code/mounirassif/pytorch-card-classifier-accuracy-96)

## Project Structure

```
card_classification
├── src
│   ├── dataset.py       # Contains the PlayingCardDataset class for loading and processing the dataset.
│   ├── model.py         # Defines the SimpleCardClassifier class for the model architecture.
│   ├── train.py         # Implements the training loop for the model.
│   ├── evaluate.py      # Functions for evaluating model performance on validation and test datasets.
│   └── utils.py         # Utility functions for image preprocessing and visualization.
├── notebooks
│   └── card_classification.ipynb  # Jupyter notebook for the entire workflow of the project.
├── requirements.txt     # Lists the dependencies required for the project.
└── README.md            # Documentation for the project.
```

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using the following command:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Use the Jupyter notebook `notebooks/card_classification.ipynb` to run the entire workflow, including data loading, model training, evaluation, and visualization of results.
 
2. Modify the parameters in the notebook as needed to suit your specific requirements.
 
3. Alternatively, run the notebook directly online:

👉 Use the Kaggle link to explore and run it on Kaggle : [👉 Run on Kaggle](https://www.kaggle.com/code/mounirassif/pytorch-card-classifier-accuracy-96)

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

