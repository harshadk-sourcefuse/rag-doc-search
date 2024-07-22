from enum import Enum


class RetrieverSearchType(Enum):
    """
    Enum representing the search types available for a vectors retriever.

    Attributes:
    - `mmr`: Refers to the search type 'Maximum marginal relevance retrieval'.
    - `similarity`: Refers to the search type 'Similarity search retrieval'.
    - `similarity_score_threshold`: Refers to the search type 'Similarity score threshold retrieval'.
    """

    mmr = "mmr"
    similarity = "similarity"
    similarity_score_threshold = "similarity_score_threshold"
