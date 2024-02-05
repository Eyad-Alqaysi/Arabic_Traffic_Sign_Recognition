
# Arabic Traffic Sign Recognition
Traffic Sign Recognition Project This repository contains the implementation of our traffic sign recognition method, including Jupyter notebooks for demonstration, training scripts, and configuration files. It's designed to provide a comprehensive overview of our approach to recognizing traffic signs using machine learning techniques.

Project Structure:

| Component                        | Description                                                               |
|----------------------------------|---------------------------------------------------------------------------|
| `.ipynb_checkpoints/`            | Directory for Jupyter notebook autosave files.                            |
| `models/`                        | Contains saved models and weights.                                        |
| `ATSR_our_method.ipynb`           | Jupyter notebook with our traffic sign recognition method.                |
| `test_on_test_data.py`                   | Script for preparing and loading the test data.                           |
| `train.py`                       | Main training script for the model.                                       |
| `test_on_train_data.py`                  | Script for preparing and loading the training data.                       |
| `test_on_val_data.py`                    | Script for preparing and loading the validation data.                     |


Setup:

To get started with this project, clone this repository to your local machine:

`https://github.com/Eyad-Alqaysi/Arabic_Traffic_Sign_Recognition.git`

`cd Arabic_Traffic_Sign_Recognition`

Ensure you have the necessary dependencies installed. While the specific requirements may vary, a typical setup might look like this:

`pip install -r requirements.txt`

Usage:

To explore our traffic sign recognition method, start with the TSR_our_method.ipynb Jupyter notebook:

`jupyter notebook TSR_our_method.ipynb`

For training the model with your data, run:

`python train.py`

Contributing:

We welcome contributions to improve this project! Whether it's by reporting issues, adding new features, or improving documentation, your help is appreciated. Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

License:

This project is licensed under the MIT License - see the LICENSE.md file for details.
