import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()
# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



df = pd.read_csv("Dataset\Mental_Health_FAQ.csv", na_filter=False)
df = df[['Questions', 'Answers']]

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


df['tokenized Questions'] = df['Questions'].apply(preprocess_text)

# Create a corpus by flattening the preprocessed questions
corpus = df['tokenized Questions'].tolist()

# Vectorize corpus
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(corpus)
# TDIDF is a numerical statistic used to evaluate how important a word is to a document in a collection or corpus. 
# The TfidfVectorizer calculates the Tfidf values for each word in the corpus and uses them to create a matrix where each row represents a document and each column represents a word. 
# The cell values in the matrix correspond to the importance of each word in each document.



def get_response(user_input):
    global most_similar_index
    global similarity_scores
    global most_similar_index
    
    user_input_processed = preprocess_text(user_input) # ....................... Preprocess the user's input using the preprocess_text function

    user_input_vector = tfidf_vectorizer.transform([user_input_processed])# .... Vectorize the preprocessed user input using the TF-IDF vectorizer

    similarity_scores = cosine_similarity(user_input_vector, X) # .. Calculate the score of similarity between the user input vector and the corpus (df) vector

    most_similar_index = similarity_scores.argmax() # ..... Find the index of the most similar question in the corpus (df) based on cosine similarity

    return df['Answers'].iloc[most_similar_index] # ... Retrieve the corresponding answer from the df DataFrame and return it as the chatbot's response

# create greeting list 
greetings = ["Hey There.... I am a creation of Ehiz Danny Agba Coder.... How can I help",
            "Hi Human.... How can I help",
            'Twale baba nla, wetin dey happen nah',
            'How far Alaye, wetin happen'
            "Good Day .... How can I help", 
            "Hello There... How can I be useful to you today",
            "Hi GomyCode Student.... How can I be of use"]

exits = ['thanks bye', 'bye', 'quit', 'exit', 'bye bye', 'close']
farewell = ['Thanks....see you soon', 'Babye, See you soon', 'Bye... See you later', 'Bye... come back soon']

random_farewell = random.choice(farewell) # ---------------- Randomly select a farewell message from the list
random_greetings = random.choice(greetings) # -------- Randomly select greeting message from the list

# Test your chatbot
while True:
    user_input = input("You: ")
    if user_input.lower() in exits:
        print(f"\nChatbot: {random_farewell}!")
        break
    if user_input.lower() in ['hi', 'hello', 'hey', 'hi there']:
        print(f"\nChatbot: {random_greetings}!")
    else:   
        response = get_response(user_input)
        print(f"\nChatbot: {response}")

tfidf_vectorizer = TfidfVectorizer()
xtrain = tfidf_vectorizer.fit_transform(df['tokenized Questions'])
# Xtrain is the preprocessed questions 



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

# Transform the Y 
df['Answers_ID'] = le.fit_transform(df['Answers'])
df.head()

ytrain = df['Answers_ID'].values
# ytrain is the transformed Answers 

# ---------------------------------- STREAMLIT IMPLEMENTATION --------------------------------------
import streamlit as st


st.title("CHATBOT MACHINE.")
st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

quit_sentences = ['quit', 'bye', 'Goodbye', 'exit']

history = []

st.markdown('<h3>Quit Words are: Quit, Bye, Goodbye, Exit</h3>', unsafe_allow_html = True)

# Get the user's question    
user_input = st.text_input(f'Input your response')
if user_input not in quit_sentences:
    if st.button("Submit"):
        # Call the chatbot function with the question and display the response
        response = get_response(user_input)
        st.write("Chatbot: " + response)

        # Create a history for the chat
        history.append(('User: ', user_input))
        history.append(('Bot: ', get_response(user_input)))
else:
    st.write('Bye')

st.markdown('<hr><hr>', unsafe_allow_html= True)
st.sidebar.subheader('Chat History')

chat_history_str = '\n'.join([f'{sender} {message}' for sender, message in history])

st.sidebar.text_area('Conversation', value=chat_history_str, height=300)