# chatbot.py

import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import random

# Download tokenizer data once (comment after 1st run)
nltk.download('punkt')

# Step 1: Intents data
intents = {
    "greeting": {
        "examples": ["Hi", "Hello", "Hey", "Howdy", "What's up?"],
        "responses": ["Hello!", "Hey there!", "Hi, how can I help you today?"]
    },
    "goodbye": {
        "examples": ["Bye", "See you", "Goodbye", "Catch you later"],
        "responses": ["Goodbye!", "See you later!", "Have a nice day!"]
    },
    "thanks": {
        "examples": ["Thanks", "Thank you", "Thanks a lot"],
        "responses": ["You're welcome!", "No problem!", "Happy to help!"]
    },
    "name": {
        "examples": ["What is your name?", "Who are you?", "Your name?"],
        "responses": ["I'm ChatBot!", "People call me ChatBot."]
    },
    "age": {
        "examples": ["How old are you?", "What is your age?"],
        "responses": ["I'm timeless!", "I don't have an age."]
    },

    # 🌱 Career Guidance
    "career_options": {
        "examples": ["What career options do I have?", "Suggest a career path", "Career advice"],
        "responses": ["You can explore careers like Data Scientist, Software Engineer, or UX Designer depending on your interests."]
    },
    "skills_required": {
        "examples": ["What skills do I need to become a Data Scientist?", "Skills for AI", "How to get into ML?"],
        "responses": ["To become a Data Scientist, you need Python, SQL, statistics, machine learning, and good communication skills."]
    },
    "college_courses": {
        "examples": ["Which courses are good for AI?", "What should I study for data science?", "Course recommendation"],
        "responses": ["Look into courses like BSc in CS, MSc in Data Science, or online programs from Coursera, edX, or Udemy."]
    },

    # 🏥 Healthcare Q&A
    "symptoms_check": {
        "examples": ["I have a headache and fever", "I'm coughing a lot", "My throat hurts"],
        "responses": ["That could be a viral infection. Please consult a doctor if symptoms persist."]
    },
    "diet_advice": {
        "examples": ["What diet should I follow?", "Healthy diet tips", "Suggest food for weight loss"],
        "responses": ["Include more veggies, fruits, and lean proteins. Avoid processed food and drink plenty of water."]
    },
    "exercise_tips": {
        "examples": ["How should I start exercising?", "Easy workouts for beginners", "Give me some fitness tips"],
        "responses": ["Start with walking, light yoga, or home workouts. Consistency is more important than intensity at first."]
    },

    # 💼 Interview / Job Help
    "interview_tips": {
        "examples": ["Tips for job interviews", "How to crack an interview?", "Interview preparation"],
        "responses": ["Practice common questions, know your resume, and show confidence! Research the company beforehand."]
    },
    "resume_help": {
        "examples": ["How to make a resume?", "Resume tips", "What to add in my CV?"],
        "responses": ["Keep it 1 page, add skills, education, and projects. Use a clean format and tailor it to each job."]
    },

    # 🧠 Motivation / Mental Well-being
    "motivation": {
        "examples": ["I feel low", "I need motivation", "I'm stressed"],
        "responses": ["Take a deep breath. You’re doing great. Progress, not perfection 💪"]
    },
    "study_tips": {
        "examples": ["How to focus on studying?", "I get distracted", "Tips for studying better"],
        "responses": ["Try the Pomodoro technique, turn off notifications, and study in short bursts with breaks."]
    }
}

# Step 2: Preprocessing
stemmer = PorterStemmer()

def tokenize_and_stem(sentence):
    tokens = nltk.word_tokenize(sentence.lower())
    stems = [stemmer.stem(token) for token in tokens]
    return " ".join(stems)

# Step 3: Prepare training data
sentences = []
labels = []

for intent, data in intents.items():
    for example in data["examples"]:
        sentences.append(example)
        labels.append(intent)

processed_sentences = [tokenize_and_stem(sentence) for sentence in sentences]

# Step 4: Vectorize sentences
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(processed_sentences)

# Step 5: Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

# Step 6: Train model
model = LogisticRegression()
model.fit(X, y)

# Step 7: Chatbot response function
def chatbot_response(user_input):
    user_input_processed = tokenize_and_stem(user_input)
    user_vec = vectorizer.transform([user_input_processed])
    pred = model.predict(user_vec)
    intent = label_encoder.inverse_transform(pred)[0]
    
    response = random.choice(intents[intent]["responses"])
    return response

# ✅ Step 8: Exportable response function for web use
def get_response(msg):
    return chatbot_response(msg)
