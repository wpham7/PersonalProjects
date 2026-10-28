# Embeddings & Vector DB — Progress

Rules: NumPy only until milestone 4. No sklearn, Chroma, or FAISS.

## Phase 1 – Embeddings
- [x] 1. Bag-of-words
  - [x] tokenize
  - [x] build_vocab (first-appearance order)
  - [x] vectorize
  - [x] cosine similarity
  - [x] similarity matrix
- [x] 2. TF-IDF weighting
  - [x] document_frequency
  - [x] idf
  - [x] tfidf_vectorize
  - [x] compare vs bag-of-words
- [x] 3. Word2vec skip-gram on a small corpus
  - [x] skipgram_pairs (center, context) with a window
  - [x] init_weights (W_in, W_out)
  - [x] softmax (stable: subtract max before exp)
  - [x] forward pass
  - [x] loss
  - [x] train_step (gradients + update)
  - [x] training loop
  - [x] most_similar(word)
- [x] 4. Compare against sentence-transformers
  - [x] install + embed docs with a pretrained model (venv in .venv/)
  - [x] compare similarity matrices: BoW vs TF-IDF vs pretrained

## Phase 2 – Vector DB
- [ ] 5. Brute-force store: add(id, vec, metadata), search(q, k)
  - [ ] VectorStore class + add
  - [ ] search(q, k)
- [ ] 6. Persistence to disk
- [ ] 7. IVF index with own k-means; recall@k vs brute force
- [ ] 8. HNSW (stretch); plot speed vs recall

## Phase 3 – Apply
- [ ] 9. Chunk and ingest notes, then RAG with an LLM

## Current task
5a. New file `vectordb.py`: `VectorStore` class with `__init__` and `add(id, vec, metadata)`

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
- `()` calls a function, `[]` indexes a list/array/dict
- BoW/TF-IDF only match exact tokens: `jump` ≠ `jumps` (no partial credit for similar spelling or meaning)
- Softmax overflow: shift scores by the max *before* exp (and subtract in the right direction)
- `np.multiply` (slot-by-slot) vs `@`/`np.dot` (matrix multiplication)
- `x -= y` already means `x = x - y` (right side is just the change); `x - y` alone is thrown away
- center vs context index: the truth/error is about the context (neighbor) word
- `append` (adds one item, even a whole list) vs `extend` (adds each item of a list)
- `.sort()` (in place, returns None) vs `sorted()` (returns a new list)
- Reusing a variable name (loop var shadowing a parameter) loses the original value
- venv: packages installed in .venv are only visible when VS Code / terminal use that interpreter
