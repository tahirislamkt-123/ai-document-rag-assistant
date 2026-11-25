class FaissRetriever:
    MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        documents,
        model_id=MODEL_ID,
        local_files_only=False,
        model=None,
        index=None,
    ):
        if not documents:
            raise ValueError("At least one document is required")

        import faiss
        import numpy as np

        self.documents = tuple(dict(document) for document in documents)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                model_id,
                local_files_only=local_files_only,
            )
        self.model = model
        self._faiss = faiss
        self._numpy = np

        if index is None:
            vectors = self.model.encode_document(
                [document["text"] for document in self.documents],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vectors = np.asarray(vectors, dtype="float32")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
        elif index.ntotal != len(self.documents):
            raise ValueError("FAISS index and document count do not match")
        self.index = index

    @classmethod
    def load(cls, documents, index_path, model):
        import faiss

        return cls(
            documents,
            model=model,
            index=faiss.read_index(str(index_path)),
        )

    def save(self, index_path):
        self._faiss.write_index(self.index, str(index_path))

    def search(self, question, top_k=3):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        query_vector = self.model.encode_query(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return self.search_vector(query_vector, top_k)

    def search_vector(self, query_vector, top_k=3):
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_vector = self._numpy.asarray(
            query_vector,
            dtype="float32",
        )
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[0] != 1:
            raise ValueError("search_vector expects exactly one query vector")
        scores, positions = self.index.search(
            query_vector,
            min(top_k, len(self.documents)),
        )

        hits = []
        for score, position in zip(scores[0], positions[0]):
            document = dict(self.documents[int(position)])
            document.setdefault(
                "document_id",
                document.get("source", document.get("id")),
            )
            document.setdefault("chunk_id", document["id"])
            document["score"] = float(score)
            hits.append(document)
        return hits
