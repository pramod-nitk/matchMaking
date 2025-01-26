# main.py
import os
import logging
import argparse
# Local imports
from src.utils import load_config, setup_logger
from src.data_ingestion import split_pdf, pdf_to_text_pymupdf
from src.preprocessing import clean_resume_text
from src.extraction import (
    get_server_health, 
    post_completion, 
    extract_json_sections, 
    parse_json_sections,
    parse_resume_file,
    parse_resume_file_2
)
from src.embedding import EmbeddingModel, ingest_embeddings_to_faiss
from src.matching import create_candidate_profile, search_candidates
import faiss
import numpy as np
import pickle
import yaml
import json

def task_split_and_clean(config_path: str):
    """
    1) Splits PDF and cleans resume text.
    Saves cleaned resumes into a folder or in memory for next step.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)

    input_pdf_path = config["paths"]["input_pdf_path"]
    output_folder = config["paths"]["output_folder"]
    pdf2text = config["paths"]["pdf2text"]
    os.makedirs(output_folder, exist_ok=True)

    # Split PDF
    logger.info(f"Splitting PDF: {input_pdf_path}")
    split_pdf(input_pdf_path, output_folder)

    # Clean text
    # 4. Convert each PDF to text and clean
    all_files = os.listdir(output_folder)
    resume_dict = {}
    for pdf_file in all_files:
        if pdf_file.endswith(".pdf"):
            full_path = os.path.join(output_folder, str(pdf_file))
            pdf_text = pdf_to_text_pymupdf(full_path)
            resume_dict[pdf_file] = clean_resume_text(pdf_text)

            with open(os.path.join(pdf2text, str(pdf_file.split(".pdf")[0])+".txt"), "w") as file:
                file.write(resume_dict[pdf_file])

def get_schema(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as file:
        schema = json.load(file)  # Load the JSON inside the 'with' block
    return schema

def task_extract_data_and_generate(config_path: str, file_name: str):
    """
    2) Loads cleaned resumes from pickle, extracts structured data using LLM,
       and converts them into a list of summary sentences for each resume.
    Saves structured info + sentences to a pickle.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    output_folder = config["paths"]["pdf2text"]
    candidates_profile = config["paths"]["candidatesProfile"]
    llm_server_url = config["llm_server"]["base_url"]
    all_files = os.listdir(output_folder)
    if file_name is not None:
        all_files = [file_name]
    cleaned_resume_data = {}
    for txt_file in all_files:
        if txt_file.endswith(".txt"):
            full_path = os.path.join(output_folder, txt_file)
            with open(full_path, "r") as file:
                text_data = file.read()
            cleaned_resume_data[txt_file] = text_data

    # Check LLM server health
    health = get_server_health(llm_server_url)
    if health.get("status") != "ok":
        logger.error("LLM server not ready.")
        return

    schema = get_schema("config/schema.json")

    print(schema)
    for filename, text in cleaned_resume_data.items():
        user_input = f"""You are an expert in extracting structured information from unstructured text.
                 Process the following resume text and extract key information such as 
                 Name, 
                 Date of Birth, 
                 Gender, 
                 Email, 
                 Phone Number, 
                 Address, 
                 Skills: 
                     1. technical skills <mention all technical skills from skills, projects, internships etc> 
                     2. other skills, 
                 Education 
                     1. pre-degree with CGPA or percentages
                     2. bachelors with CGPA or percentages
                     3. masters (if available) with CGPA or percentages, 
                 Extra-Curricular Activities, 
                 Projects 
                     1. detail or name of projects 
                     2. skills used in projects, 
                Internships 
                    1.company 
                    2.year 
                    3.skills
                References 
                    1.name 
                    2.designation. 
                    
                Please strictly follow this JSON structure ({schema}). If a field does not exist, use "Not Found" or an empty array/list:
                 ### Now Process This Resume:
                 ### Now Process This Resume:
                {text}"""
        response_str = post_completion(llm_server_url, "", user_input)

        with open(os.path.join(candidates_profile, str(filename)+".txt"), "w") as file:
                file.write(response_str) 


def JSON_EXTRACT(config_path: str, file_name: str):
        logger = logging.getLogger(__name__)
        config = load_config(config_path)
        candidates_profile = config["paths"]["candidatesProfile"]
        search_index = config["paths"]["search_index"]
        # Extract data + generate sentences
        structured_info = {}
        sentences_dict = {}
        all_files = os.listdir(candidates_profile)
        if file_name is not None:
            all_files = [file_name]

        failed_json = []
        for txt_file in all_files:
            if txt_file.endswith(".txt"):
                full_path = os.path.join(candidates_profile, txt_file)
                with open(full_path, "r") as file:
                    text_data = file.read()

                sections = extract_json_sections(text_data)
                if not sections:
                    logger.warning(f"No JSON found for {txt_file}")
                    failed_json.append(full_path)
                    continue
                
                try:
                    parsed = parse_json_sections(sections)[0]
                    structured_info[txt_file] = parsed

                    # Create a textual summary or "profile sentence" from parsed data
                    # This is your "generate sentences for each resume" part
                    summary_text = create_candidate_profile([parsed])
                    sentences_dict[txt_file] = summary_text

                except Exception as e:
                    logger.error(f"Parsing JSON for {txt_file} failed: {e}")

        failed_files = []
        for file in failed_json:
            # Skip if not a .txt file
            if not file.endswith(".txt"):
                continue
            data = None
            parse_success = False
            # 1) First try parse_resume_file
            try:
                data = parse_resume_file(file)
                parse_success = True
            except ValueError as e:
                print(f"parse_resume_file failed for {file}: {e}")

            # 2) Fallback: try parse_resume_file_2
            if not parse_success:
                try:
                    data = parse_resume_file_2(file)
                    parse_success = True
                except ValueError as e:
                    print(f"parse_resume_file_2 failed for {file}: {e}")
                    failed_files.append(file)

            if parse_success:
                if data is None:
                    pass
                else:
                    structured_info[file] = data
                    summary_text = create_candidate_profile(data)  # <-- Adjust if 'parsed' means 'data'
                    sentences_dict[file] = summary_text
                    print(f"\nParsed data from {file}:\n{data}\n")
        
        results_pickle_path = os.path.join(search_index, "structured_resumes.pkl")
        with open(results_pickle_path, "wb") as f:
            pickle.dump({
                "structured_info": structured_info,
                "sentences_dict": sentences_dict
            }, f)

        logger.info(f"Structured data & sentences saved to: {results_pickle_path}")


def task_generate_embeddings(config_path: str):
    """
    3) Loads the summary sentences and generates embeddings for each sentence,
       then saves them to a file for indexing later.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    output_folder = config["paths"]["search_index"]
    embd_folder = config["paths"]["embeddings"]
    results_pickle_path = os.path.join(output_folder, "structured_resumes.pkl")
    if not os.path.isfile(results_pickle_path):
        logger.error("No structured resumes found. Run 'extract_data' first.")
        return

    with open(results_pickle_path, "rb") as f:
        data = pickle.load(f)

    sentences_dict = data["sentences_dict"]  # filename -> summary_text
    print(sentences_dict)
    filenames = list(sentences_dict.keys())
    sentences = list(sentences_dict.values())

    # Initialize embedding model
    model_name = config["embedding"]["model_name"]
    embedding_model = EmbeddingModel(model_name=model_name)
    embeddings = embedding_model.encode(sentences)

    # Save embeddings and metadata
    embeddings_pickle_path = os.path.join(embd_folder, "embeddings.pkl")
    with open(embeddings_pickle_path, "wb") as f:
        pickle.dump({
            "filenames": filenames,
            "sentences": sentences,
            "embeddings": embeddings
        }, f)
    
    logger.info(f"Embeddings saved to: {embeddings_pickle_path}")



def task_index_in_faiss(config_path: str):
    """
    4) Reads embeddings.pkl, indexes them in FAISS,
       and saves the FAISS index to disk.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    output_folder = config["paths"]["embeddings"]
    faiss_idx_path = config["paths"]["faissindex"]
    embeddings_pickle_path = os.path.join(output_folder, "embeddings.pkl")
    if not os.path.isfile(embeddings_pickle_path):
        logger.error("No embeddings found. Run 'generate_embeddings' first.")
        return

    with open(embeddings_pickle_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"]
    filenames = data["filenames"]

    # Create FAISS index
    index = ingest_embeddings_to_faiss(embeddings)

    # Save the index
    import faiss
    faiss_index_path = os.path.join(faiss_idx_path, "candidates.index")
    faiss.write_index(index, faiss_index_path)
    logger.info(f"FAISS index saved to: {faiss_index_path}")

    # Also save a map from index -> filenames
    map_pickle_path = os.path.join(faiss_idx_path, "index_map.pkl")
    with open(map_pickle_path, "wb") as f:
        pickle.dump({
            "filenames": filenames
        }, f)
    logger.info(f"Index map saved to: {map_pickle_path}")


def task_run_api(config_path: str):
    """
    5) Loads FAISS index & index_map, then interacts with user to get a
       job description and prints top matched candidates.
       (Or you can spin up a web service here, e.g., using FastAPI.)
    """
    import faiss
    config = load_config(config_path)
    faiss_idx_path = config["paths"]["faissindex"]

    # Load index
    faiss_index_path = os.path.join(faiss_idx_path, "candidates.index")
    index_map_path = os.path.join(faiss_idx_path, "index_map.pkl")

    index = faiss.read_index(faiss_index_path)
    with open(index_map_path, "rb") as f:
        index_map = pickle.load(f)

    filenames = index_map["filenames"]

    # Optionally, load embedding model
    model_name = config["embedding"]["model_name"]
    embedding_model = EmbeddingModel(model_name=model_name)

    # Example: ask user for job description
    print("Enter a job description:")
    job_description = input("> ")

    # Compute embedding
    job_embedding = embedding_model.encode([job_description]).astype("float32")

    # Search top 5
    distances, indices = index.search(job_embedding, k=5)
    print("Top 5 matches:")
    for rank, i in enumerate(indices[0]):
        filename = filenames[i]
        dist = distances[0][rank]
        print(f"  {rank+1}. {filename} (distance={dist:.4f})")

    print("Done.")


def parse_args():
    setup_logger(logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Main script with function-based tasks.")
    parser.add_argument(
        "--function",
        required=True,
        choices=["split_pdf", "extract_data", "json_extract", "generate_embeddings", "index_embeddings", "run_api"],
        help="Which function to run."
    )
    parser.add_argument(
        "--file",
        default=None,  # If file is not provided, we'll treat it as None
        help="File name; if omitted, passes None."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML config file."
    )
    return parser.parse_args()

def main():
    print("inside main")
    setup_logger(logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    logger.info(f"Running function: {args.function}")
    logger.info(f"File argument is: {args.file}")
    print(args)

    if args.function == "split_pdf":
        task_split_and_clean(args.config)
    elif args.function == "extract_data":
        task_extract_data_and_generate(args.config, args.file)
    elif args.function =="json_extract":
        JSON_EXTRACT(args.config, args.file)
    elif args.function == "generate_embeddings":
        task_generate_embeddings(args.config)
    elif args.function == "index_embeddings":
        task_index_in_faiss(args.config)
    elif args.function == "run_api":
        task_run_api(args.config)
    else:
        logger.error("Unknown command.")


if __name__ == "__main__":
    main()






# def main():
#     # 1. Setup logging
#     setup_logger(logging.INFO)
#     logger = logging.getLogger(__name__)
#     logger.info("Starting student-job-match pipeline.")

#     # 2. Load configuration
#     config = load_config("config/config.yaml")
#     input_pdf_path = config["paths"]["input_pdf_path"]
#     output_folder = config["paths"]["output_folder"]
#     candidatesProfile = config["paths"]["candidatesProfile"]
#     faissindex = config["paths"]["faissindex"]
#     pdf2text = config["paths"]["pdf2text"]
#     embeddings_path = config["paths"]["embeddings"]
#     # 3. Split the PDF into individual resumes
#     split_pdf(input_pdf_path, output_folder)

#     # 4. Convert each PDF to text and clean
#     all_files = os.listdir(output_folder)
#     resume_dict = {}
#     for pdf_file in all_files:
#         if pdf_file.endswith(".pdf"):
#             full_path = os.path.join(output_folder, pdf_file)
#             pdf_text = pdf_to_text_pymupdf(full_path)
#             resume_dict[pdf_file] = clean_resume_text(pdf_text)

#             with open(os.path.join(pdf2text, str(pdf_file.split(".")[0])+".txt"), "w") as file:
#                 file.write(resume_dict[pdf_file])

    # # 5. Check server health & extract structured information
    # health = get_server_health(base_url)
    # if health.get("status") != "ok":
    #     logger.error("LLM server is not ready. Exiting.")
    #     return

    # processed_information = []
    # for name, clean_text in resume_dict.items():
    #     user_input = (
    #         "Extract key structured information (Name, DOB, Gender, Email, etc.) from the text below. "
    #         "Output in JSON. If data is missing, mark as 'Not Found'.\n\n"
    #         f"Resume:\n{clean_text}"
    #     )
    #     assistant_response = post_completion(base_url, "", user_input)
    #     processed_information.append((name, assistant_response))

    # # 6. Parse the LLM JSON output
    # processed_resume_data_json = {}
    # for filename, response_str in processed_information:
    #     sections = extract_json_sections(response_str)
    #     if sections:
    #         try:
    #             parsed = parse_json_sections(sections)
    #             # You might want to handle multiple JSON blocks; for now, assume first
    #             name_in_json = parsed[0].get("Name", filename)
    #             processed_resume_data_json[name_in_json] = parsed
    #         except Exception as e:
    #             logger.error(f"Could not parse JSON for {filename}: {e}")

    # # 7. Create textual profiles & build embeddings
    # logger.info("Generating textual profiles for each candidate...")
    # profiles = []
    # for key, val in processed_resume_data_json.items():
    #     profile_text = create_candidate_profile(val)  # or build your own
    #     profiles.append(profile_text)

    # # 8. Encode profiles and create FAISS index
    # embedding_model = EmbeddingModel()
    # profile_embeddings = embedding_model.encode(profiles)
    # index = ingest_embeddings_to_faiss(profile_embeddings)

    # # 9. Example job matching
    # example_job_description = "Looking for a candidate skilled in Python, Java, and Mathematics."
    # job_embedding = embedding_model.encode([example_job_description]).astype('float32')
    # results = search_candidates(index, processed_resume_data_json, job_embedding)

    # logger.info("Top matched candidates (distance):")
    # for candidate_name, dist in results:
    #     logger.info(f"{candidate_name} => {dist}")

    # logger.info("Pipeline finished successfully.")

# if __name__ == "__main__":
#     main()
