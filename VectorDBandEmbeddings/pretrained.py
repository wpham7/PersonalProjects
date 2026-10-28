from sentence_transformers import SentenceTransformer
from bow import build_vocab, vectorize, idf, tfidf_vectorize, similarity_matrix

model = SentenceTransformer('all-MiniLM-L6-v2')

if __name__ == "__main__":
    docs = [
    "the cat ran",
    "the dog ran",
    "the human jump",
    "human boy jumps",
    "dad grabs is pumpkin",
    "dad is grabbing pumpkins"
    ]

# What I coded aka BoW and IDF
    words = build_vocab(docs)
    idf_weights = idf(docs, words)

    BoW_vectors = []
    IDF_vectors = []
    for doc in docs:
        BoW_vectors.append(vectorize(doc, words))
        IDF_vectors.append(tfidf_vectorize(doc, words, idf_weights))

# pretrained model / SentenceTransformer
    embeddings = model.encode(docs)


    print(f"similarity matrix of BoW: \n{similarity_matrix(BoW_vectors).round(2)}")
    print(f"similarity matrix of IDF: \n{similarity_matrix(IDF_vectors).round(2)}")
    print(f"similarity matrix of pretrained: \n{similarity_matrix(embeddings).round(2)}")

