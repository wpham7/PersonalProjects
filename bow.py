import re
import numpy as np

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

if __name__ == "__main__":
    #print(tokenize("Hello World! This is a test."))
    print(build_vocab(["Hello World!", "This is a test."]))