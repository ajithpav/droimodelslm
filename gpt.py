import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Load models and tokenizers
model_name = "droidal_finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# File paths
WEBSITE_TEXT_PATH = "output/droidal_website_text.txt"
COMBINED_TEXT_PATH = "output/droidal_combined_text.txt"
CLEANED_TEXT_PATH = "output/cleaned_droidal_text.txt"

# Expanded predefined responses
predefined_responses = {
    "Who is the CEO of Droidal?": "Inger Sivanthi is the Chief Executive Officer of Droidal.",
    "What does Droidal do?": "Droidal provides automation solutions using advanced AI and RPA technologies.",
    "contact": "You can reach Droidal at contact@droidal.com or call us at our office.",
    "location": "Droidal's headquarters is located in [Location].",
    "services": "Droidal offers AI solutions, RPA implementation, and digital transformation services."
}

class TextProcessor:
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
        
        self._prepare_embeddings()

    def _load_text_file(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        return ""

    def _prepare_embeddings(self):
        # Split texts into sentences and create embeddings
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
        """Process text into clean, meaningful sentences."""
        # Split text into sentences
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        # Clean sentences
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = self._clean_sentence(sentence)
            if cleaned and len(cleaned.split()) >= 5:  # Ensure meaningful content
                cleaned_sentences.append(cleaned)
                
        return cleaned_sentences

    def _clean_sentence(self, sentence):
        """Clean individual sentences."""
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'[^\w\s.,!?-]', '', sentence)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        # Capitalize first letter
        cleaned = cleaned.strip().capitalize()
        if cleaned and not cleaned.endswith(('.', '!', '?')):
            cleaned += '.'
        return cleaned

    def find_relevant_response(self, query, threshold=0.6):
        """Find the most relevant response from all available text sources."""
        query_embedding = sentence_model.encode([query])
        best_response = None
        highest_similarity = threshold

        # Search in all text sources
        sources = [
            (self.website_embeddings, self.website_sentences),
            (self.combined_embeddings, self.combined_sentences),
            (self.cleaned_embeddings, self.cleaned_sentences)
        ]

        for embeddings, sentences in sources:
            if embeddings is not None and len(sentences) > 0:
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                max_sim = np.max(similarities)
                if max_sim > highest_similarity:
                    highest_similarity = max_sim
                    best_response = sentences[np.argmax(similarities)]

        return best_response

def clean_response(text):
    """Clean and format the response text."""
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove any HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Capitalize first letter and add period if missing
    text = text.strip().capitalize()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

# Initialize text processor
text_processor = TextProcessor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json
        user_input = data.get("message", "").strip()

        # Handle empty input
        if not user_input:
            return jsonify({"response": "I didn't receive any input. Could you please ask your question?"})

        # Check for keywords in predefined responses
        for key, response in predefined_responses.items():
            if key.lower() in user_input.lower():
                return jsonify({"response": response})

        # Search for relevant response using semantic similarity
        relevant_response = text_processor.find_relevant_response(user_input)
        if relevant_response:
            if len(relevant_response.split()) > 40:
                relevant_response = summarizer(relevant_response, max_length=40, min_length=15, do_sample=False)[0]["summary_text"]
            return jsonify({"response": clean_response(relevant_response)})

        # Use the fine-tuned model as fallback
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
        
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")  # Log the error
        return jsonify({
            "response": "I apologize, but I'm having trouble processing your request. Could you please rephrase your question?"
        })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)