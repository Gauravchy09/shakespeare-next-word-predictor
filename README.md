Shakespeare Next Word Predictor ✒️

A Streamlit web app that uses a deep learning model (LSTM) to predict the next word in a sequence, trained on Shakespeare's Hamlet.

Demo

A quick look at the app predicting the next word.

Features

Real-time Prediction: Enter a sequence of words and get the next word instantly.

Deep Learning Model: Powered by a Keras LSTM model trained on Shakespearean text.

Interactive UI: A simple and thematic web interface built with Streamlit.

Thematic Design: Styled to match the "Bard of Avon" theme.

Technology Stack

Python: Core programming language.

TensorFlow & Keras: For building and training the LSTM model.

NLTK: Used to source the Hamlet dataset (nltk.corpus.shakespeare) for training.

Pickle: For saving and loading the Keras Tokenizer object.

Streamlit: For creating and serving the interactive web app.

Numpy: For numerical operations and handling predictions.

Project Structure

.
├── app.py                   # The Streamlit application script
├── next_word_lstm.h5      # The trained Keras LSTM model
├── tokenizer.pickle       # The saved Keras Tokenizer
├── requirements.txt         # Project dependencies
├── .gitignore               # Files to be ignored by Git
└── README.md                # You are here!


Installation & Setup

Clone the repository:

git clone [https://github.com/Gauravchy09/shakespeare-next-word-predictor.git](https://github.com/Gauravchy09/shakespeare-next-word-predictor.git)
cd shakespeare-next-word-predictor


Create a virtual environment (recommended):

python -m venv env
# On Windows:
.\env\Scripts\activate
# On MacOS/Linux:
source env/bin/activate


Install the dependencies:

pip install -r requirements.txt


How to Run

Ensure your trained model (next_word_lstm.h5) and tokenizer (tokenizer.pickle) are in the root directory.

Run the Streamlit app:

streamlit run app.py


Open your browser and navigate to http://localhost:8501.

Credits

This project was created by Gaurav Choudhary.
