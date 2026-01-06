# Product Review Sentiment Analysis with Deep Learning 💬

This project builds a deep learning model to perform multi-class sentiment analysis on customer reviews. Using a Recurrent Neural Network (LSTM/GRU), the model classifies Amazon Fine Food Reviews into **positive**, **neutral**, or **negative** categories. It covers the complete end-to-end Natural Language Processing (NLP) workflow, from raw text preprocessing to model training and in-depth evaluation.

-----

## Table of Contents

  - [Key Features](https://www.google.com/search?q=%23key-features)
  - [Tech Stack](https://www.google.com/search?q=%23tech-stack)
  - [Dataset](https://www.google.com/search?q=%23dataset)
  - [Project Workflow](https://www.google.com/search?q=%23project-workflow)
  - [Installation & Usage](https://www.google.com/search?q=%23installation--usage)
  - [Results](https://www.google.com/search?q=%23results)
  - [Future Improvements](https://www.google.com/search?q=%23future-improvements)
  - [License](https://www.google.com/search?q=%23license)

-----

## Key Features ✨

  * **Multi-Class Classification**: Classifies sentiment into three distinct categories, going beyond a simple positive/negative binary model.
  * **Recurrent Neural Networks**: Implements and compares LSTM (Long Short-Term Memory) and GRU (Gated Recurrent Unit) layers to capture context in sequential text data.
  * **Word Embeddings**: Uses the Keras `Embedding` layer to learn dense vector representations of words, allowing the model to understand semantic relationships.
  * **End-to-End NLP Pipeline**: Demonstrates the full process of cleaning text, tokenizing, padding sequences, building, training, and evaluating a neural network.
  * **In-Depth Evaluation**: Goes beyond simple accuracy by generating detailed classification reports and confusion matrices to diagnose the model's performance on an imbalanced dataset.

-----

## Tech Stack 🛠️

  * **TensorFlow / Keras**: For building and training the deep learning model.
  * **Scikit-learn**: For data splitting and generating evaluation metrics.
  * **Pandas**: For data loading and manipulation.
  * **NLTK**: For text preprocessing, specifically for removing stopwords.
  * **Matplotlib & Seaborn**: For data visualization and plotting results.
  * **NumPy**: For numerical operations.

-----

## Dataset 📦

This project uses the **Amazon Fine Food Reviews** dataset from Kaggle. It contains over 500,000 food reviews from Amazon users, including the review text and a corresponding star rating (1 to 5).

  - **Source**: [Kaggle Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
  - **Preprocessing**: The 1-5 star ratings were mapped to three sentiment classes:
      - ⭐ & ⭐⭐  →  **Negative (0)**
      - ⭐⭐⭐      →  **Neutral (1)**
      - ⭐⭐⭐⭐ & ⭐⭐⭐⭐⭐ →  **Positive (2)**

-----

## Project Workflow ⚙️

The project follows a standard machine learning workflow:

1.  **Data Loading & Cleaning**: The raw CSV data is loaded into a Pandas DataFrame. The review text is cleaned by converting to lowercase, removing punctuation and numbers, and filtering out common English stopwords.
2.  **Tokenization & Sequencing**: The cleaned text is converted into sequences of integers using the Keras `Tokenizer`, where each integer represents a unique word.
3.  **Sequence Padding**: All sequences are padded or truncated to a uniform length to be fed into the RNN.
4.  **Model Building**: A Keras `Sequential` model is constructed with the following architecture:
      * `Embedding` Layer: To learn word vectors.
      * `LSTM` or `GRU` Layer: To process the sequence and learn contextual patterns.
      * `Dense` Layer: The final output layer with a `softmax` activation for multi-class classification.
5.  **Training**: The model is compiled with the `adam` optimizer and `sparse_categorical_crossentropy` loss function. It's then trained on the training dataset while being monitored on a validation set to check for overfitting.
6.  **Evaluation**: The trained model's performance is evaluated on the unseen test set. Results are visualized by plotting training history, a classification report, and a confusion matrix.

-----

## Results 📈

The model achieves a validation accuracy of **[Your Accuracy Here]%**. The evaluation shows that the model performs well, especially in distinguishing positive and negative reviews, though it sometimes confuses neutral reviews with positive ones due to the dataset's imbalance.

#### Classification Report

```
              precision    recall  f1-score   support

    negative       0.85      0.82      0.83     16395
     neutral       0.65      0.55      0.60      8528
    positive       0.93      0.96      0.94     88787

    accuracy                           0.90    113710
   macro avg       0.81      0.78      0.79    113710
weighted avg       0.89      0.90      0.90    113710
```


-----

## Future Improvements 🔮

  * **Use Pre-trained Embeddings**: Integrate pre-trained word embeddings like **GloVe** or **Word2Vec** to potentially improve performance by leveraging knowledge from a larger corpus.
  * **Bidirectional LSTMs**: Implement a `Bidirectional` wrapper on the LSTM layer to allow the network to learn from both past (left-to-right) and future (right-to-left) context.
  * **Hyperparameter Tuning**: Use tools like KerasTuner or Optuna to systematically find the optimal set of hyperparameters (e.g., embedding dimension, LSTM units, dropout rate).
  * **Handle Class Imbalance**: Employ techniques such as class weighting during training or oversampling methods like SMOTE to improve the model's performance on minority classes (especially 'neutral').

-----

## License 📜

This project is licensed under the MIT License. See the `LICENSE` file for more details.
