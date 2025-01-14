import nltk
from nltk.chat.util import Chat, reflections
import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

# Function for Named Entity Recognition (NER) using spaCy
def analyze_text(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

# Define the pattern-response pairs
pairs = [
    (r'Hi|Hello|Hey', ['Hello!', 'Hi there!']),
    (r'How are you?', ['I am fine, thank you!', 'Doing great, how about you?']),
    (r'What is your name?', ['I am a chatbot created by CodTech.', 'You can call me CodBot.']),
    (r'What is today\'s date?', ['Let me check the calendar for you!', 'Today is 11th January, 2025.']),
    (r'Bye|Exit', ['Goodbye!', 'See you later!'])
]

# Main chatbot function
def chatbot():
    print("Hello! I am here to help you. Type 'Bye' to exit.")
    
    # Start conversation loop
    while True:
        user_input = input("You: ")
        
        # Check if user wants to exit the conversation
        if user_input.lower() == "bye":
            print("Goodbye! Have a great day!")
            break
        
        # Use spaCy for NER
        entities = analyze_text(user_input)
        if entities:
            print(f"Named entities detected: {entities}")
        
        # Initialize NLTK chatbot
        chat = Chat(pairs, reflections)
        
        # Get response from chatbot based on input
        response = chat.respond(user_input)
        
        if response:
            print(f"Bot: {response}")
        else:
            print("Bot: I'm sorry, I didn't understand that.")

if __name__ == "__main__":
    chatbot()
