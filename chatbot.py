from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# This function stores everything the chatbot "remembers"
def memory_list():
    memory = [
        "I like learning",
        "I love coding",
        "My favorite sport is football",
        "I enjoy pizza",
        "I am a chatbot"
    ]
    return memory

# This is the brain of the chatbot
# It tries to find the most similar memory to what the user says
def brain(text):
    memory = memory_list()

    # Combine old memory + new user message for comparison
    texts = memory + [text]

    # Convert text into numbers (TF-IDF vectors)
    vec = TfidfVectorizer(stop_words='english')
    emb = vec.fit_transform(texts)

    # Separate user input from memory
    user_vec = emb[-1]
    mem_vec = emb[:-1]

    # Compare similarity between user input and each memory sentence
    scores = cosine_similarity(user_vec, mem_vec)

    # Find the best match
    best_score = scores.max()
    best_index = scores.argmax()

    # If nothing is similar enough, say we don’t know
    if best_score < 0.2:
        return "I do not know yet"

    # Otherwise return the closest memory
    return memory[best_index]

# This runs the chatbot in a loop
def run():
    while True:
        text = input("user: ")
        if text:
            reply = brain(text)
            print(reply)

run()