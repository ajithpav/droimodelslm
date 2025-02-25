#slm ai model chat bot

# Droidal Chatbot

Droidal Chatbot is an AI-powered conversational assistant that leverages a fine-tuned language model to provide intelligent and context-aware responses. The chatbot supports text-based interactions and voice recognition, integrating NLP techniques and pre-trained transformers.

## Features

- **Natural Language Processing (NLP):** Uses a fine-tuned transformer model for generating responses.
- **Predefined Responses:** Handles common queries with predefined answers.
- **Semantic Search:** Finds relevant answers from processed text data using sentence embeddings.
- **Summarization:** Condenses long responses for better readability.
- **Speech Recognition & Synthesis:** Supports voice input and generates spoken responses.
- **Optimized Caching & Multithreading:** Improves performance using caching and parallel processing.

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- Pip
- Virtual Environment (optional but recommended)

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/droidal-chatbot.git
   cd droidal-chatbot
   ```
2. Create a virtual environment (optional):
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Chatbot

1. Start the Flask application:
   ```sh
   python app.py
   ```
2. Open a browser and navigate to `http://127.0.0.1:5000/`.

### API Endpoints

- `POST /chat` - Process user messages and return AI-generated responses.
  ```json
  {
    "message": "What services does Droidal offer?"
  }
  ```
  **Response:**
  ```json
  {
    "response": "Droidal provides automation solutions using advanced AI and RPA technologies."
  }
  ```
- `POST /voice` - Processes voice input and returns a transcribed response.
- `GET /health` - Returns the status of the application.

## Project Structure
```
├── app.py                  # Main Flask application
├── templates/              # HTML templates for frontend
├── static/                 # Static files (CSS, JS, audio)
├── output/                 # Processed text data
├── requirements.txt        # Dependencies
├── README.md               # Documentation
```

## Dependencies

This project uses the following libraries:
- `Flask` - Web framework for handling API requests.
- `transformers` - NLP models for text generation and summarization.
- `sentence-transformers` - Semantic search and similarity detection.
- `speechrecognition` - Speech-to-text processing.
- `pyttsx3` - Text-to-speech synthesis.

## Future Enhancements

- Implement real-time WebSocket communication.
- Support multiple languages.
- Improve model efficiency with quantization.
- Enhance voice synthesis with neural TTS models.

## Contributing

Contributions are welcome! Feel free to fork the repository and submit a pull request with improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

