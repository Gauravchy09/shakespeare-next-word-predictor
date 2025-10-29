import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import time

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="The Bard's Next Word",
    page_icon="✒️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- 2. Load Model and Tokenizer (Cached) ---
# Use st.cache_resource to load these heavy assets only once
@st.cache_resource
def load_assets():
    """Loads the trained model and tokenizer from disk."""
    try:
        model = load_model('next_word_lstm.h5', compile=False)
    except OSError:
        st.error("Model file (next_word_lstm.h5) not found. Please ensure it's in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

    try:
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    except FileNotFoundError:
        st.error("Tokenizer file (tokenizer.pickle) not found. Please ensure it's in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading tokenizer: {e}")
        return None, None, None
        
    # Retrieve max_sequence_len from the model's input shape
    # (model.input_shape[1] is the padded length, which is max_len - 1)
    max_sequence_len = model.input_shape[1] + 1
    return model, tokenizer, max_sequence_len

model, tokenizer, max_sequence_len = load_assets()

# --- 3. Prediction Function ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    """Tokenizes, pads, and predicts the next word."""
    token_list = tokenizer.texts_to_sequences([text])[0]
    
    # Handle empty or OOV (Out-of-Vocabulary) input
    if not token_list:
        return None 
        
    # Truncate from the left if text is too long
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
        
    # Pad the sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    
    # Predict probabilities
    predicted_probs = model.predict(token_list, verbose=0)[0]
    
    # Get the index of the word with the highest probability
    predicted_word_index = np.argmax(predicted_probs)
    
    # Convert index back to word
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None # Should not be reached if vocab is correct

# --- 4. Sidebar Content ---
st.sidebar.title("About This App")
st.sidebar.image("https://images.pexels.com/photos/356043/pexels-photo-356043.jpeg", 
                 caption="Trained on the works of the Bard.")
st.sidebar.info(
    """
    This app uses a **Long Short-Term Memory (LSTM)** neural network 
    to predict the next word in a sequence.
    
    **Dataset:** *Hamlet* by William Shakespeare (from `nltk.corpus.shakespeare`)
    
    **Model:** A `tensorflow.keras` Sequential model.
    """
)

st.sidebar.header("How It Works")
st.sidebar.markdown(
    """
    1.  **Enter Text:** You provide a starting phrase.
    2.  **Tokenize:** The text is converted into a sequence of numbers (tokens) based on the vocabulary.
    3.  **Pad/Truncate:** The sequence is resized to match the model's expected input length (`max_sequence_len - 1`).
    4.  **Predict:** The LSTM model predicts the probability for *every* word in its vocabulary.
    5.  **Reveal:** The word with the highest probability is chosen and displayed.
    """
)

# --- ADDED THIS LINE ---
st.sidebar.markdown("---")
st.sidebar.caption("Made by Gaurav Choudhary")
# ---------------------


# --- 5. Main Page Interface ---
st.title("The Bard's Next Word ✒️")
st.markdown("An LSTM Oracle trained on Shakespeare's *Hamlet*. Speak thy line and see what follows.")

# Create a form for the input and button
with st.form(key="prediction_form"):
    input_text = st.text_input(
        "Enter thy prose:", 
        "To be or not to"
    )
    
    submit_button = st.form_submit_button(
        label="Foretell the Word", 
        type="primary",
        use_container_width=True
    )

# --- 6. Prediction Logic and Output ---
if submit_button and model:
    if not input_text.strip():
        st.warning("Pray, enter some words first, forsooth!")
    else:
        # Show a spinner while predicting
        with st.spinner("The model divines..."):
            time.sleep(0.5) # Add a small delay for effect
            next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
        
        if next_word:
            st.subheader("Thus, the line continues:")
            
            # Custom HTML for a "parchment" styled output box
            output_html = f"""
            <div style="
                background-color: #FFF8E7; 
                border: 2px solid #8B4513; 
                border-radius: 10px; 
                padding: 20px; 
                font-family: 'Georgia', serif; 
                font-size: 1.25em;
                line-height: 1.6;
            ">
                {input_text} <strong style="color: #B22222; text-decoration: underline;">{next_word}</strong>
            </div>
            """
            st.markdown(output_html, unsafe_allow_html=True)
        else:
            st.error("Alas, the Muses are silent. I cannot find a word. (Perhaps thy words are unknown to me?)")

elif not model or not tokenizer:
    st.error("Application is not ready. Model or tokenizer files are missing.")