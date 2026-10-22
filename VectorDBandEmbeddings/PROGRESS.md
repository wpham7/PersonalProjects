# Embeddings & Vector DB — Progress

Rules: NumPy only until milestone 4. No sklearn, Chroma, or FAISS.

## Phase 1 – Embeddings
- [ ] 1. Bag-of-words
  - [x] tokenize
  - [x] build_vocab (first-appearance order)
  - [x] vectorize
  - [x] cosine similarity
  - [ ] similarity matrix
- [ ] 2. TF-IDF weighting
- [ ] 3. Word2vec skip-gram on a small corpus
- [ ] 4. Compare against sentence-transformers

## Phase 2 – Vector DB
- [ ] 5. Brute-force store: add(id, vec, metadata), search(q, k)
- [ ] 6. Persistence to disk
- [ ] 7. IVF index with own k-means; recall@k vs brute force
- [ ] 8. HNSW (stretch); plot speed vs recall

## Phase 3 – Apply
- [ ] 9. Chunk and ingest notes, then RAG with an LLM

## Current task
1e. `similarity_matrix(vectors)` → n×n NumPy array of pairwise cosine similarities

## Concepts I struggled with
- Python syntax habits carried over from Java
- Referencing a method without calling it (`string.split` vs `string.split()`)
- Assigning to a loop variable doesn't change the list (strings are immutable; build a new list)
- Regex quantifiers (`?` / `*` / `+`) and character classes `[...]`
- Generator `(...)` vs list `[...]` comprehensions (generators are single-use)
- Why a vocab is a dict (word → index), not a set or list
- Missing dict key raises KeyError (no Java-style null); use `key in d`
- Looping over the right thing (letters vs tokens vs vocab) — recurring in 1a–1c
- `len(v)` = number of slots, not vector length (that's `np.linalg.norm`); `or` vs `|`
