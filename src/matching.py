# src/matching.py
import logging

logger = logging.getLogger(__name__)

def create_candidate_profile(candidate_data):
    """
    Example of how to generate a profile text from candidate data.
    Modify to fit your needs.
    """
    skills = []
    # For illustration, just gather 'Skills'
    for index in candidate_data.keys():
        if index=='Skills':
            if isinstance(candidate_data[index], dict):
                if 'technical_skills' in candidate_data[index].keys():
                    skills.extend(candidate_data[index]['technical_skills'])
                elif 'other_skills' in candidate_data[index].keys():
                    skills.extend(candidate_data[index]['other_skills'])

    profile_sentence = (
        f"Candidate has these skills: {', '.join(skills) if skills else 'Not Found'}."
    )
    return profile_sentence

def search_candidates(index, candidate_dict, job_description_embedding, top_k=5):
    """
    Searches the FAISS index for the most relevant candidates.
    Returns a list of (candidate_name, distance) for the top_k matches.
    """
    distances, indices = index.search(job_description_embedding, k=top_k)
    # Retrieve keys in the same order they were added
    keys = list(candidate_dict.keys())
    results = [(keys[i], distances[0][rank]) for rank, i in enumerate(indices[0])]
    return results
