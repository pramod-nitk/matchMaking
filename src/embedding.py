"""
This module provides a simple embedding model wrapper and a helper function
to ingest embeddings into a FAISS index.
"""

import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

logger = logging.getLogger(__name__)

class EmbeddingModel: # pylint: disable=too-few-public-methods
    """
    A wrapper around a SentenceTransformer embedding model.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        init function
        """
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Generate embeddings for the given texts.

        :param texts: A string or a list of strings to encode.
        :type texts: Union[str, List[str]]
        :return: Numpy array of embeddings.
        :rtype: np.ndarray
        """
        return self.model.encode(texts)

def ingest_embeddings_to_faiss(embeddings):
    """
    Ingest embeddings(numpy array) into a FAISS IndexFlatL2 index.
    :param embeddings: A numpy array of shape (N, D) containing embeddings,
                       where N is number of embeddings and D is the embedding dimension.
    :type embeddings: np.ndarray
    :return: A FAISS IndexFlatL2 object containing the ingested embeddings.
    :rtype: faiss.IndexFlatL2
    """
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    # Pylint may throw a false-positive for FAISS usage; ignore it:
    index.add(embeddings) # pylint: disable=no-value-for-parameter
    return index
