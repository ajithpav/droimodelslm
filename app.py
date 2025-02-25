import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import speech_recognition as sr
import pyttsx3
import threading
import queue
from functools import lru_cache
import tempfile
import wave
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Constants
WEBSITE_TEXT_PATH = "output/droidal_website_text.txt"
COMBINED_TEXT_PATH = "output/droidal_combined_text.txt"
CLEANED_TEXT_PATH = "output/droidal_cleaned_text.txt"
AUDIO_OUTPUT_DIR = "static/audio"
Path(AUDIO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Load models and tokenizers
model_name = "droidal_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Predefined responses
predefined_responses = {
    "Who is the CEO of Droidal?": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "ceo of droidal": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "droidal ceo": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "Droidal of ceo": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "Who is the CEO of Droidal?": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "What does Droidal do?": "Droidal provides automation solutions using advanced AI and RPA technologies.",
    "contact": "You can reach Droidal at contact@droidal.com or call us at our office.",
    "location": "Droidal's headquarters is located in [Location].",
    "services": "Droidal offers AI solutions, RPA implementation, and digital transformation services."
}

class VoiceProcessor:
    """Handle voice-related processing with optimized settings"""
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.engine = pyttsx3.init()
        
        # Optimize voice recognition
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True
        
        # Optimize text-to-speech
        self.engine.setProperty('rate', 180)
        self.engine.setProperty('volume', 0.9)
        
        # Setup voices
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)  # Index 0 for male, 1 for female

    def recognize_speech(self, audio_data):
        """Convert speech to text"""
        try:
            return self.recognizer.recognize_google(audio_data)
        except (sr.UnknownValueError, sr.RequestError):
            return None

    def text_to_speech(self, text, output_path):
        """Convert text to speech and save to file"""
        try:
            self.engine.save_to_file(text, output_path)
            self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            return False

class TextProcessor:
    """Handle text processing and response generation"""
    def __init__(self):
        self.website_text = self._load_text_file(WEBSITE_TEXT_PATH)
        self.combined_text = self._load_text_file(COMBINED_TEXT_PATH)
        self.cleaned_text = self._load_text_file(CLEANED_TEXT_PATH)
        
        self.website_embeddings = None
        self.combined_embeddings = None
        self.cleaned_embeddings = None
        
        self.website_sentences = None
        self.combined_sentences = None
        self.cleaned_sentences = None
        
        self.voice_processor = VoiceProcessor()
        
        # Initialize processing
        self._prepare_embeddings()
        self.response_cache = {}

    def _load_text_file(self, file_path):
        """Load and read text file"""
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def _prepare_embeddings(self):
        """Prepare text embeddings for faster processing"""
        self.website_sentences = self._process_text(self.website_text)
        self.combined_sentences = self._process_text(self.combined_text)
        self.cleaned_sentences = self._process_text(self.cleaned_text)
        
        if self.website_sentences:
            self.website_embeddings = sentence_model.encode(self.website_sentences)
        if self.combined_sentences:
            self.combined_embeddings = sentence_model.encode(self.combined_sentences)
        if self.cleaned_sentences:
            self.cleaned_embeddings = sentence_model.encode(self.cleaned_sentences)

    def _process_text(self, text):
        """Process text into clean sentences"""
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        cleaned_sentences = []
        
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and len(cleaned.split()) >= 5:
                cleaned_sentences.append(cleaned)
                
        return cleaned_sentences

    def _clean_sentence(self, sentence):
        """Clean individual sentence"""
        cleaned = re.sub(r'[^\w\s.,!?-]', '', sentence)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip().capitalize()
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        return cleaned

    @lru_cache(maxsize=1000)
    def find_relevant_response(self, query, threshold=0.6):
        """Find relevant response with caching and parallel processing"""
        query_embedding = sentence_model.encode([query])
        best_response = None
        highest_similarity = threshold

        def process_source(embeddings, sentences):
            if embeddings is not None and len(sentences) > 0:
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                max_sim = np.max(similarities)
                if max_sim > highest_similarity:
                    return (max_sim, sentences[np.argmax(similarities)])
            return (highest_similarity, None)

        # Process sources in parallel
        threads = []
        results = queue.Queue()
        
        sources = [
            (self.website_embeddings, self.website_sentences),
            (self.combined_embeddings, self.combined_sentences),
            (self.cleaned_embeddings, self.cleaned_sentences)
        ]

        for embeddings, sentences in sources:
            thread = threading.Thread(
                target=lambda q, emb, sent: q.put(process_source(emb, sent)),
                args=(results, embeddings, sentences)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        while not results.empty():
            sim, response = results.get()
            if sim > highest_similarity and response:
                highest_similarity = sim
                best_response = response

        return best_response

def process_chat_input(text_processor, user_input):
    """Process text input and generate response"""
    # Check predefined responses
    for key, response in predefined_responses.items():
        if key.lower() in user_input.lower():
            return clean_response(response)

    # Get relevant response
    relevant_response = text_processor.find_relevant_response(user_input)
    if relevant_response:
        if len(relevant_response.split()) > 40:
            relevant_response = summarizer(relevant_response, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
        return clean_response(relevant_response)

    # Use model as fallback
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=500,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True,
        num_return_sequences=1
    )
    
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = clean_response(response_text)
    
    if len(response_text.split()) > 40:
        response_text = summarizer(response_text, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
    
    return response_text

def clean_response(text):
    """Clean and format response text"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    text = text.strip().capitalize()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# Initialize processors
text_processor = TextProcessor()
response_queue = queue.Queue()

@app.route('/')
def home():
    """Render home page"""
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    """Handle text chat requests"""
    try:
        data = request.json
        user_input = data.get("message", "").strip()

        if not user_input:
            return jsonify({"response": "I didn't receive any input. Could you please ask your question?"})

        response = process_chat_input(text_processor, user_input)
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Chat Error: {str(e)}")
        return jsonify({
            "response": "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"
        })

@app.route("/voice-chat", methods=["POST"])
def voice_chat():
    """Handle voice chat requests"""
    try:
        # Get audio file
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"})

        # Process audio in separate thread
        def process_audio():
            try:
                # Convert audio to format recognized by speech_recognition
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    audio_file.save(temp_audio.name)
                    with sr.AudioFile(temp_audio.name) as source:
                        audio_data = text_processor.voice_processor.recognizer.record(source)

                # Convert speech to text
                text_input = text_processor.voice_processor.recognize_speech(audio_data)
                if not text_input:
                    response_queue.put({"error": "Could not understand audio"})
                    return

                # Generate text response
                text_response = process_chat_input(text_processor, text_input)

                # Generate voice response
                response_filename = f"response_{int(time.time())}.mp3"
                response_path = os.path.join(AUDIO_OUTPUT_DIR, response_filename)
                
                if text_processor.voice_processor.text_to_speech(text_response, response_path):
                    response_queue.put({
                        "text": text_response,
                        "voice_url": f"/static/audio/{response_filename}"
                    })
                else:
                    response_queue.put({
                        "text": text_response,
                        "error": "Voice generation failed"
                    })

            except Exception as e:
                print(f"Audio Processing Error: {str(e)}")
                response_queue.put({"error": "Audio processing failed"})

        # Start processing in background
        thread = threading.Thread(target=process_audio)
        thread.start()
        thread.join(timeout=10)

        # Get response
        if not response_queue.empty():
            return jsonify(response_queue.get())
        else:
            return jsonify({"error": "Processing timeout"})

    except Exception as e:
        print(f"Voice Chat Error: {str(e)}")
        return jsonify({"error": "Voice processing error"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)