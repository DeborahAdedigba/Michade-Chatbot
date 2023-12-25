import json
import pickle
import random
import tensorflow as tf
import nltk
import numpy as np
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from nltk.stem import WordNetLemmatizer

app = Flask(__name__)

# Disable TensorFlow progress reports
nltk.download('punkt')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Load the initial intents data from a JSON file
intents = json.load(open('michade_farms.json'))

# Load the previously pickled words and classes lists
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
try:
    model = load_model("my_model.keras")
except Exception as e:
    print(f"Error loading the model: {e}")
    # Handle the error or exit the program

# Constants
GIBBERISH_THRESHOLD = 0.1
FALLBACK_THRESHOLD = 0.2

# Initialize the WordNetLemmatizer for lemmatization
lemmatizer = WordNetLemmatizer()


# Define the ChatContext class to manage conversation context
class ChatContext:
    def __init__(self):
        self.current_intent = None
        self.user_input_history = []
        self.last_response = None
        self.user_name = None
        self.user_location = None
        self.user_preferences = {}


# Function to handle gibberish responses when the chatbot doesn't understand the user
def handle_gibberish_response():
    return "Mich Bot: I'm sorry, but I couldn't understand your input. Please try again."


# Function to handle fallback responses when the chatbot doesn't understand the user
def handle_fallback_response(context):
    response = "Mich Bot: Sorry, I don't understand. Would you like to talk to the support team?"
    user_response = request.form['user_message'].lower()
    if user_response in ["yes", "yeah", "ok", "sure"]:
        # Provide contact information for the support team
        support_email = "info@michadefarms.com.ng"
        support_phone = "+2348037149761"
        response = (f"Mich Bot: Sure! You can reach our support team via email at {support_email} "
                    f"or by phone at {support_phone}.")
    elif user_response in ["no", "nope"]:
        response = "Mich Bot: No problem. Would you like to ask me any other questions?"
    else:
        response = "Mich Bot: I'm sorry, please respond with 'yes' or 'no'."

    # Update context with the fact that the chatbot didn't understand the user's input
    context.last_response = "I don't understand. Please try again."
    return response


# Function to clean up a sentence by tokenizing and lemmatizing the words
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


# Function to convert a sentence into a bag of words representation
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


# Function to predict the class (intent) of a given sentence using the trained model
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list


# Function to get a random response for a given intent from the intents data
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


# Function to handle user input and display responses
@app.route('/')
def home():
    return render_template('templates/chatbot_interface.html')



@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['user_message'].lower()

    # Create context
    context = ChatContext()

    if user_message in ["quit", "exit"]:
        return jsonify({'bot_response': "Mich Bot: Goodbye! Chatbot is now exiting."})

    # Predict the intent and get the response
    ints = predict_class(user_message)

    # Identify gibberish responses
    if not ints or float(ints[0]['probability']) < GIBBERISH_THRESHOLD:
        response = handle_gibberish_response()
    else:
        # Identify fallback responses
        if float(ints[0]['probability']) < FALLBACK_THRESHOLD:
            response = handle_fallback_response(context)
        else:
            res = get_response(ints, intents)
            response = f"Mich Bot: {res}"

    return jsonify({'bot_response': response})


if __name__ == '__main__':
    app.run(debug=True)
