import re
import numpy as np

#BoW = Bag of Words

#Takes each sentence and grabs individual words. 
def tokenize(text: str):
    return re.findall(r"[\w']+", text.lower()) #Regex translation: [\w'] every word including apostrophes, + 1 or more

#Uses tokenize() to create a dict of all known words. 
def build_vocab(docs):
    vocab = dict()
    for doc in docs:
        tokens = tokenize(doc)
        for word in tokens:
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab

###
# Every time the incoming question / prompt has a word that matches to a 
# a word in our vocab dict, add 1 to the corresponding slot in vectors.
###
def vectorize(text, vocab):
    vectors = np.zeros(len(vocab))
    words = tokenize(text)
    for word in words:
        if word in vocab:
            vectors[vocab[word]] += 1
    return vectors

# Takes two vectors and sees if they point in the same direction aka
# how similar are these two sentences?
def cosine_similarity(a, b):
    if np.linalg.norm(a) and np.linalg.norm(b) != 0:
        #(a*b) / (length of a * length of b)
        similarity = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return similarity
    else: return 0

#Creates a matrix of similarities between different vectors/sentences
def similarity_matrix(vectors):
    n = len(vectors)
    sim_matrix = np.zeros((n, n))
    for row in range(n):
        for column in range(n):
            sim_matrix[row, column] = cosine_similarity(vectors[row], vectors[column])

    return sim_matrix

"""
Term Frequency - Inverse Document Frequency

Description:
Between sentences "The dog" and "The cat" there's 50% similarity
Even though the word 'the' is common, so we need to give it less weight

Term Frequency = how often a word appears in doc
Inverse Document Frequency = how rare the word is in the docs
"""

# Check how many docs each word appears in
def document_frequency(docs, vocab):
    frequency = np.zeros(len(vocab))

    for doc in docs:
        tokens = set(tokenize(doc))
        for word in tokens:
            if word in vocab:
                frequency[vocab[word]] += 1
    return frequency

# Now turn the document counts into rarity scores
def idf(docs,vocab):
    idf = np.log(len(docs) / document_frequency(docs, vocab))
    return idf

def tfidf_vectorize(text, vocab, idf_weights):
    return np.multiply(vectorize(text, vocab), idf_weights)

"""
Word2Vec: The idea that words used in similar places mean similar things

Example: Jump vs Jumps should be valued similarly but previous code
deems them completely separate words

So Word2Vec learns a small vector for each word so that words with similar
neighbors get similar vectors.
Example: 'cat' and 'dog' might show up next to the 'ran'. so they should
be valued similarly
"""

# Makes (word, nearby word) pairs for word2vec training.
def skipgram_pairs(tokens, window):
    pairs = []
    for i in range(len(tokens)):
        for position in range(i-window, i+window+1):
            if position != i and position >= 0 and position < len(tokens):
                pairs.append((tokens[i], tokens[position]))
    return pairs

# Create arrays for weights of each word (these are random to start), dim = dimensions (how many numbers each word's vector has)
def init_weights(vocab_size, dim):
    W_in = np.random.randn(vocab_size, dim) * 0.01
    W_out = np.random.randn(dim, vocab_size) * 0.01
    return W_in, W_out

# Turns raw scores into probabilities that sum to 1 (higher score = higher probability).
def softmax(scores):
    normalized_scores = scores - np.max(scores)
    return np.exp(normalized_scores) / np.sum(np.exp(normalized_scores))

# Model makes it's first guess
def forward(center_idx, W_in, W_out):
    current_vec = W_in[center_idx]
    scores = np.dot(current_vec, W_out)
    probabilities = softmax(scores)
    return current_vec, probabilities

# Gives a number for how wrong the guess was
def loss(probs, context_idx):
    return -np.log((probs[context_idx]))

# Now we can implement training by nudging the initial weights, lr = learning rate (how big each nudge is)
def train_step(center_idx, context_idx, W_in, W_out, lr):
    h, probs = forward(center_idx, W_in, W_out)

    error = probs.copy()
    error[context_idx] -= 1

    #Note: grad short for gradient aka technical name for which direction to nudge and how much
    # Nudge for W_out
    grad_out = np.outer(h, error)

    # Nudge for h
    grad_h = np.dot(W_out, error)

    # Apply nudges
    W_out -= lr * grad_out
    W_in[center_idx] -= lr * grad_h

    return loss(probs, context_idx)

# Train over a corpus, epoch = one full pass through training pair
def train(docs, dim, window, epochs, lr):
    vocab = build_vocab(docs)
    W_in, W_out = init_weights(len(vocab), dim)

    pairs = []
    index_pairs = []
    for doc in docs:
        tokens = tokenize(doc)
        pairs.extend(skipgram_pairs(tokens, window))

    for pair in pairs:
        center, context = pair
        index_pairs.append((vocab[center], vocab[context]))

    for epoch in range(epochs):
        total = 0
        for center,context in index_pairs:
            step_loss = train_step(center, context, W_in, W_out, lr)
            total += step_loss
        if epoch % 20 == 0:
            print(f"epoch {epoch}: loss {total / len(pairs)}")

    return W_in, vocab

# Find the word whose learned vectors are closest to the given wor
def most_similar(word, W_in, vocab, k):
    word_vector = W_in[vocab[word]]
    scored_list = []
    for other, idx in vocab.items():
        if other != word:
            score = (cosine_similarity(W_in[idx], word_vector))
            scored_list.append((other, score))
    sorted_list = sorted(scored_list, key=lambda pair: pair[1], reverse=True)
    return sorted_list[:k]



            
            


if __name__ == "__main__":
    docs = [
        "the cat ran",
        "the dog ran",
        "the human jump",
        "human boy jumps",
        "dad grabs is pumpkin",
        "dad is grabbing pumpkins"
    ]


    words = build_vocab(docs)
    idf_weights = idf(docs, words)

    BoW_vectors = []
    IDF_vectors = []
    for doc in docs:
        BoW_vectors.append(vectorize(doc, words))
        IDF_vectors.append(tfidf_vectorize(doc, words, idf_weights))

    print(f"similarity matrix of BoW: \n{similarity_matrix(BoW_vectors).round(2)}")
    print(f"similarity matrix of IDF: \n{similarity_matrix(IDF_vectors).round(2)}")
