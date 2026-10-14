# 🌍 Language Identification Using LSTM

### A Deep Learning Approach for Multilingual Text Classification

<p align="center">
  <strong>Natural Language Processing | Deep Learning | LSTM | Python</strong>
</p>

---

## 📌 Project Overview

Language Identification Using LSTM is a Deep Learning project that automatically identifies the language of a given text using a Long Short-Term Memory (LSTM) neural network.

Language identification is an important Natural Language Processing (NLP) task used in multilingual applications, text processing systems, translation services, and language-aware software.

This project uses text preprocessing, tokenization, sequence padding, and an LSTM-based classification model to learn language-specific patterns and classify text into one of 22 languages.

The model was trained and evaluated using a balanced subset of a Kaggle Language Identification dataset.

### Key Highlights

- Supports classification across 22 languages.
- Uses an LSTM neural network for sequential text learning.
- Implements text preprocessing and tokenization.
- Uses stratified train-test splitting.
- Evaluates performance using multiple classification metrics.
- Visualizes classification results using a confusion matrix.
- Supports predictions on user-provided text.

---

## 🎯 Project Objectives

The primary objectives of this project are:

1. Develop a deep learning model for multilingual language identification.
2. Preprocess text data and convert text into numerical sequences.
3. Train an LSTM network to learn language-specific patterns.
4. Classify input text into one of 22 supported languages.
5. Evaluate model performance using accuracy, precision, recall, and F1-score.
6. Analyze classification performance using a confusion matrix.
7. Test the trained model using unseen text inputs.

---

## 🛠️ Technologies and Tools

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network implementation |
| Pandas | Dataset loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data splitting and model evaluation |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Google Colab | Model development and training |
| Kaggle | Dataset source |

---

## 📂 Dataset Description

The project uses a **Language Identification dataset from Kaggle** containing text samples labeled with their corresponding languages.

The dataset was cleaned and preprocessed before training. A balanced subset was selected to ensure that every language contributed the same number of samples.

### Dataset Summary

| Attribute | Description |
|---|---|
| Dataset | Kaggle Language Identification |
| Original dataset after preprocessing | 21,858 samples |
| Number of languages | 22 |
| Samples selected per language | 600 |
| Total samples used | 13,200 |
| Training samples | 10,560 |
| Testing samples | 2,640 |
| Training-test ratio | 80:20 |
| Sampling method | Balanced sampling |
| Split method | Stratified train-test split |
| Random state | 42 |

### Supported Languages

The model was trained to identify the following 22 languages:

1. Arabic
2. Chinese
3. Dutch
4. English
5. Estonian
6. French
7. Hindi
8. Indonesian
9. Japanese
10. Korean
11. Latin
12. Persian
13. Portuguese
14. Pushto
15. Romanian
16. Russian
17. Spanish
18. Swedish
19. Tamil
20. Thai
21. Turkish
22. Urdu

*Note: Language names follow the dataset's labeling conventions. The original dataset label for Portuguese may appear as `Portugese`, and Pashto may appear as `Pushto`.*

### Dataset Distribution

| Dataset Split | Samples | Percentage |
|---|---:|---:|
| Training | 10,560 | 80% |
| Testing | 2,640 | 20% |
| **Total** | **13,200** | **100%** |

Each language contributes:

- **480 training samples**
- **120 testing samples**

Stratified splitting preserves the language distribution in both sets.

> **Dataset source:** Add the original Kaggle dataset URL here before publishing the repository.

---

## 🧠 Model Architecture

The project uses an LSTM-based neural network for multilingual text classification.

LSTM is a type of recurrent neural network designed to learn sequential dependencies in data. Its memory cell and gating mechanisms help the network retain or discard information as it processes a sequence.

### Architecture of the Original Model

The original model consists of the following layers:

| Layer | Configuration | Purpose |
|---|---|---|
| Embedding | 64-dimensional vectors | Represents tokens as dense vectors |
| LSTM | 64 units | Learns sequential text patterns |
| Dropout | 0.3 | Helps reduce overfitting |
| Dense | 22 units, Softmax | Predicts probabilities for 22 languages |

### Model Workflow

```text
                Input Text
                    |
                    v
             Text Preprocessing
                    |
                    v
                Tokenization
                    |
                    v
               Sequence Padding
                    |
                    v
              Embedding Layer
                    |
                    v
                 LSTM Layer
                    |
                    v
               Dropout Layer
                    |
                    v
           Dense Output Layer
              Softmax Activation
                    |
                    v
          Predicted Language Label
```

### Why LSTM?

LSTM was selected because text is sequential data.

The model can learn patterns associated with character or word sequences and use those patterns to distinguish between languages.

Its gating mechanisms help manage information over a sequence, making it suitable for sequence-based classification tasks.

However, LSTM is not necessarily the best architecture for every language. Character-level methods and Transformer-based approaches may perform better for some multilingual text classification problems.

---

## ⚙️ Methodology

The project follows a structured deep learning workflow.

### 1. Dataset Loading

The dataset is loaded into a Pandas DataFrame.

The text and language label columns are used for preprocessing and classification.

### 2. Data Preprocessing

The text is cleaned to reduce unnecessary variation.

The preprocessing workflow includes:

- Removing unnecessary punctuation where appropriate.
- Normalizing whitespace.
- Preparing text for tokenization.
- Checking and handling missing or invalid records.

Unicode text must be preserved correctly to support languages that use different writing systems.

### 3. Balanced Sampling

A balanced subset is created by selecting 600 samples from each language.

With 22 languages:

**22 × 600 = 13,200 samples**

This provides equal representation of each language in the selected dataset.

### 4. Label Encoding

Each language label is converted into a numerical class using Scikit-learn's `LabelEncoder`.

The numerical labels are used as targets during model training.

### 5. Train-Test Splitting

The selected dataset is divided into:

- 80% training data
- 20% testing data

Stratified splitting helps maintain equal language representation across the two sets.

### 6. Tokenization

The original model uses a Keras word-level tokenizer to convert text into numerical sequences.

A vocabulary is built from the training text, and each token is mapped to an integer.

### 7. Sequence Padding

Sequences are padded or truncated to a fixed length of 100 tokens.

This allows the model to process batches of input with consistent dimensions.

### 8. Model Training

The LSTM model is trained using the following configuration:

| Parameter | Value |
|---|---|
| Embedding dimension | 64 |
| LSTM units | 64 |
| Dropout rate | 0.3 |
| Optimizer | Adam |
| Loss function | Sparse categorical cross-entropy |
| Batch size | 64 |
| Epochs | 5 |
| Output activation | Softmax |

### 9. Model Evaluation

The trained model is evaluated on the held-out test set.

Performance is measured using accuracy, precision, recall, F1-score, and a confusion matrix.

### 10. Language Prediction

The model accepts new text, applies the same preprocessing and tokenization pipeline, and predicts the most likely language among the 22 classes.

---

## 📊 Model Performance

The original word-level LSTM model achieved the following evaluation results on the test set.

| Metric | Score |
|---|---:|
| **Accuracy** | **90.45%** |
| Macro Precision | 91.25% |
| Macro Recall | 90.45% |
| Macro F1-Score | 90.60% |

### Interpretation

**Accuracy — 90.45%**

The model correctly classified approximately 90.45% of the samples in the test set.

**Macro Precision — 91.25%**

The average precision across the 22 language classes was approximately 91.25%, giving equal weight to each language.

**Macro Recall — 90.45%**

The average recall across the 22 languages was approximately 90.45%.

**Macro F1-Score — 90.60%**

The macro F1-score summarizes the balance between precision and recall across the language classes.

> **Important:** These metrics correspond to the original word-level LSTM model. They must not be presented as the results of a different model unless that model has been trained and evaluated separately.

---

## 📈 Evaluation Techniques

### 1. Accuracy

Accuracy measures the proportion of test samples classified correctly.

It provides an overall view of model performance.

### 2. Precision

Precision measures how many samples predicted as a particular language actually belong to that language.

High precision indicates fewer false-positive predictions for that class.

### 3. Recall

Recall measures how many samples belonging to a language were correctly identified.

High recall indicates fewer missed samples for that class.

### 4. F1-Score

The F1-score is the harmonic mean of precision and recall.

It is useful when both false-positive and false-negative classifications matter.

### 5. Confusion Matrix

A confusion matrix compares actual language labels with predicted labels.

It helps identify which languages the model distinguishes successfully and which languages are frequently confused.

For example, languages with overlapping characters or similar writing systems may be more difficult to distinguish.

---

## 🚀 Installation and Execution

### Prerequisites

Ensure that Python is available, or use Google Colab to run the notebook without a local Python installation.

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/language-identification-lstm.git
cd language-identification-lstm
```

Replace `YOUR_USERNAME` with your GitHub username and update the repository name if necessary.

### Step 2: Install Dependencies

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Step 3: Obtain the Dataset

Download the original Language Identification dataset from Kaggle.

Place the dataset in your project directory or upload it to Google Colab.

Update the dataset file path in the notebook to match its actual location.

### Step 4: Open the Notebook

Open the `.ipynb` notebook in Google Colab or Jupyter Notebook.

### Step 5: Execute the Cells

Run the notebook cells in order to:

1. Import the required libraries.
2. Load and preprocess the dataset.
3. Select balanced samples.
4. Encode language labels.
5. Split the dataset.
6. Tokenize and pad the text.
7. Build and train the LSTM model.
8. Evaluate model performance.
9. Generate a confusion matrix.
10. Predict languages for new text.

---

## 🔍 Example Prediction

### Input

```text
Bonjour, comment allez-vous?
```

### Expected Output

```text
Predicted Language: French
```

*This is an illustrative example. The actual prediction depends on the trained model.*

### Testing Custom Text

The notebook includes an interactive input function that accepts a sentence and returns the predicted language.

Example:

```python
text = input("Enter a sentence: ")
```

The text is passed through the same preprocessing, tokenization, and padding pipeline before the model produces its prediction.

---

## ⚠️ Limitations

Although the original model achieved 90.45% test accuracy, several limitations should be considered.

### 1. Word-Level Tokenization

The original model uses word-level tokenization.

This approach can be ineffective for languages such as Chinese, where words are not normally separated by spaces.

### 2. Similar Writing Systems

Some languages share characters or writing-system features, which can lead to confusion between language classes.

For example, Chinese and Japanese may share certain characters.

### 3. Short Text Inputs

Very short sentences may not contain enough distinctive information for reliable classification.

### 4. Unseen Vocabulary

Words that do not appear in the training vocabulary may be mapped to an out-of-vocabulary token.

This can reduce prediction quality.

### 5. Dataset Dependence

The model's performance depends on the quality, variety, and representativeness of the training data.

A balanced test set does not guarantee the same performance on naturally occurring text.

### 6. No Universal Accuracy Guarantee

The reported metrics reflect one dataset and one train-test split. Performance may differ on other datasets, writing styles, and real-world applications.

---

## 🔮 Future Enhancements

Potential improvements include:

- **Character-Level Tokenization:** Better represent languages that do not use whitespace-separated words.
- **Bidirectional LSTM:** Learn sequential patterns in both directions.
- **Hyperparameter Tuning:** Experiment with embedding dimensions, LSTM units, dropout, and learning rates.
- **Early Stopping:** Stop training when validation performance no longer improves.
- **Learning-Rate Scheduling:** Adjust the learning rate during training.
- **Transformer Models:** Compare LSTM with modern contextual language models.
- **Improved Dataset Diversity:** Add more examples covering different text lengths and writing styles.
- **Interactive Web Application:** Deploy the model using Streamlit, Flask, or FastAPI.
- **Model Export:** Save the trained model and tokenizer for reuse without retraining.

These are proposed enhancements and should not be considered implemented unless they are added and tested.

---

## 🎓 Applications

Language identification can support:

- Multilingual text-processing systems.
- Automatic language detection in user-generated content.
- Language-aware document organization.
- Preliminary routing in translation workflows.
- Text analytics and multilingual data processing.
- Natural Language Processing research and experimentation.

---

## 📚 Key Concepts Learned

Through this project, the following concepts were explored:

- Natural Language Processing
- Text preprocessing
- Tokenization and sequence padding
- Label encoding
- Stratified train-test splitting
- Word embeddings
- Recurrent Neural Networks
- Long Short-Term Memory networks
- Multiclass classification
- Model evaluation metrics
- Confusion matrix analysis
- Prediction on unseen text

---

## 👩‍💻 Author

**Evangelin Golda**

Computer Science and Engineering Student

- **GitHub:** https://github.com/evangelin-golda
- **LinkedIn:** https://www.linkedin.com/in/evangelingolda

---

## 📄 License

Unless a license is added to this repository, all rights remain with the respective copyright holders.

If you intend to distribute this project as open source, consider adding an appropriate license and verifying the dataset's terms of use.

---

## ⭐ Acknowledgements

- The creators and maintainers of the Kaggle Language Identification dataset.
- The TensorFlow and Keras teams for deep learning tools.
- The Scikit-learn community for data processing and evaluation utilities.
- The Python open-source community.

---

<p align="center">
  <strong>Built with Python, TensorFlow, and LSTM</strong>
  <br>
  Exploring multilingual text classification through Deep Learning.
</p>
