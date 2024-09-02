# Michade ChatBot

Michade ChatBot - An AI-powered chatbot for assisting with ranch-related inquiries and animal farming guidance.

## Project Overview

The Michade ChatBot is an AI-driven conversational agent designed to provide comprehensive information about Michade Farms and offer guidance for individuals interested in starting their own animal farming ventures. This chatbot leverages Natural Language Processing (NLP) and machine learning to simulate human-like interactions, making it easier for users to access relevant agricultural knowledge and farm-related services.

## Features

- **General Inquiries**: Provides details about Michade Farms, including location, services, and amenities.
- **Animal Farming Guidance**: Offers beginner-friendly advice on raising livestock such as chickens, cows, and goats.
- **Customer Support**: Answers frequently asked questions and directs users to customer service when needed.
- **Ethical AI**: The chatbot is designed with ethical considerations, ensuring user privacy, fairness, and accuracy of information.

## Installation

1. Clone the repository:

    ```
    git clone https://github.com/DeborahAdedigba/chatBOT.git
    ```

2. Run the chatbot:

    Open and run the `JYPD_training.ipynb` file to train the model.  
    Open and run the `JYPD_main.ipynb` file to start the chatbot interface.

## Usage

1. **Start the Chatbot**: After running `JYPD_main.ipynb`, a tkinter window will pop up.
2. **Interact**: You can ask questions related to Michade Farms or seek advice on starting your animal farming venture.
3. **Exit**: Type `exit` or `quit` to close the chatbot interface.

## Project Structure

- **JYPD_training.ipynb**: This notebook contains the code for training the chatbot's machine learning model.
- **JYPD_main.ipynb**: This notebook handles the chatbot's main functionalities, including user interaction and response generation.

## Libraries and Tools

- **Python**: Core programming language.
- **TensorFlow**: For training the neural network model.
- **Keras**: For building and training deep learning models.
- **Scikit-learn**: Used for various machine learning tasks like data preprocessing and model evaluation.
- **Streamlit**: For creating a simple web app interface (optional).
- **Plotly**: For creating interactive plots.
- **Matplotlib**: For data visualization.
- **Feedparser**: To handle RSS feeds.
- **tkinter**: For building the chatbot’s graphical user interface.

## Ethical Considerations

The chatbot is developed with attention to ethical AI principles:
- **Privacy**: Ensures user data is handled securely and transparently.
- **Fairness**: Strives to eliminate biases in the training data.
- **Accuracy**: Provides reliable and accurate information to users.

## Limitations

- Limited to English language interactions.
- Predefined set of intents and responses.
- May not remember context between different sessions.

## Future Improvements

- Integration of more advanced models such as BERT or GPT-3 for enhanced language understanding.
- Adding multi-language support.
- Enhancing the chatbot's memory for maintaining context across sessions.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE License.

