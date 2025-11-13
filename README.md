# DATA641 - Assignment 3

Course (PCS): DATA 641 (PCS4)
Author: Emily Hightower
Date Due: 2025.11.12

The following project and text are responses adapted from "Homework 3" by Professor Naeemul Hassan (2025, UMD).

# Project Description:
This project explores Sentiment Classification as a core Natural Language Processing (NLP) task used to categorize the emotional tone of a piece of text into classes like positive or negative. The project implements and evaluates Recurrent Neural Networks (RNN) architectures for sentiment classification, as treated as a sequence classification problem.

The project uses the IMDb Movie Review Dataset, which contains 50,000 reviews. The dataset is organized into 25,000 training and 25,000 testing examples. For each review, the dataset is labeled with the class (positive, negative).

The following code processes the text for analysis, trains a model configuration with the training dataset, and tests each model configuration with the testing dataset. The models are evaluated using accuracy, F1-score and training loss over time. The model configurations tested are as follows:
- Architecture: RNN, LSTM, Bidirectional LSTM
- Activation Function: sigmoid, relu, tanh
- Optimizer: adam, stochastic gradient descent (sgd), RMSProp
- Sequence Length: 35, 50, 100
- Stability Strategy: none, gradient clipping

The total number of model combinations is 162 models. However, LSTM and Bidirectional LSTM only use tanh as the activation function as tanh naturally scales between -1 and 1, stabilizing the cell state and final hidden output. Sigmoid is used in three other gates (input, forget, and output), and application of relu or an arbitrary activation in place of tanh causes the gating logic to break.

The code is created to run each unique model configuration as called by the driver.py code (162 models) and the model.py code is configured to ignore activation functions that are not allowed. This leads to multiple trained versions of the LSTM and BiLSTM models for the other configuration settings (as tanh is the activation function in all cases). Thus, there are 162 models trained but 90 unique model configurations.

After training the models, the models are evaluated to show F1-score, accuracy, and the training loss associated with the best and worst models.

# Program Components & Functionality

**Recommendations:**
- Save the dataset as "dataset.csv" if not using the pre-saved IMDb dataset in GitHub.
- Review the .py documents to ensure the pathways match appropriately.
- If re-implementing this code, rename, move or delete presaved processed documents (i.e., metrics, processed/, results/, preprocessing_stats, preprocessing_times).
- Use the following hardware or GPUs to run the program with similar timing.

**Hardware:**
- CPU: arm
- CPU cores: 14
- Logical processors: 14
- RAM: 48.00 GB
- GPU: None

**Libraries:**
- Core: pandas, nltk, regex, numpy
- ML/DL: torch, torchvision, torchaudio, tensorflow
- Evaluation: scikit-learn
- Utilities: tqdm, joblib

**Files (runtime in seconds):**
- hardware (0.01 seconds)
- preprocess (4.1209 seconds)
- models (*run by driver.py*)
- train (*run by driver.py*)
- evaluate (*run by driver.py*)
- util (*run by driver.py*)
- driver (6.212 hours)
*Total Runtime:* 6.214 hours

# Code Implementation:
- pip install -r requirements.txt
- python hardware.py
- python src/preprocess.py \
    --input data/dataset.csv \
    --output_dir data/processed \
    --results_file results/preprocessing_times.csv \
    --stats_file results/preprocessing_stats.csv
- python driver.py

*Example Execution: Single Model*
python src/train.py \
    --architecture LSTM \
    --activation tanh \
    --optimizer rmsprop \
    --sequence_length 100 \
    --clip \
    --epochs 5 \
    --batch_size 32 \
    --save_model

The epochs (5) and batch_size (32) are defaults. These can be set to other values as well in the single model training approach.

# Citations

Hassan N. “Homework 3.” ELMS, 2025, umd.instructure.com/courses/1395714/assignments/7388552.

Lakshmipathi N. “IMDB Dataset of 50K Movie Reviews.” Kaggle.com, 2019, www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code.

Maas, Andrew L., Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. “Learning Word Vectors for Sentiment Analysis.” Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, June 2011, Portland, Oregon, pp. 142–150. http://www.aclweb.org/anthology/P11-1015
