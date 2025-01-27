# main.py
import os
import logging
import pickle
import json
import argparse
import pandas as pd
# Local imports
from src.utils import load_config, setup_logger
from src.data_ingestion import split_pdf, pdf_to_text_pymupdf
from src.preprocessing import clean_resume_text
from src.extraction import (
    get_server_health,
    post_completion,
    # extract_json_sections,
    # parse_json_sections,
    # parse_resume_file,
    # parse_resume_file_2
)
from src.embedding import EmbeddingModel, ingest_embeddings_to_faiss
from src.matching import create_candidate_profile #, search_candidates
from src.insights import get_insights
from utils.parse_json_chatgpt import parse_resume_file_chatapi
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def process_resume(filename, text, schema, llm_server_url, candidates_profile):
    """
    Sends one resume to the LLM server and writes the response to a .txt file.
    """
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
    
    Please strictly follow this JSON structure ({schema}).
    If a field does not exist, use 'Not Found' or an empty array/list.

    ### Now Process This Resume:
    {text}"""

    response_str = post_completion(llm_server_url, "", user_input)
    output_path = os.path.join(candidates_profile, f"{filename}.txt")
    with open(output_path, "w") as f:
        f.write(response_str)
    return filename

def parallel_process_resumes(cleaned_resume_data, schema, llm_server_url, candidates_profile):
    """
    Runs the 'process_resume' function in parallel for each item in 'cleaned_resume_data'.
    """
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit tasks to the pool
        futures = {
            executor.submit(
                process_resume,
                filename,
                text,
                schema,
                llm_server_url,
                candidates_profile
            ): filename
            for filename, text in cleaned_resume_data.items()
        }

        # Retrieve results as they complete
        for future in as_completed(futures):
            fn = futures[future]
            try:
                finished_filename = future.result()  # returns filename
                results.append(finished_filename)
            except Exception as e:
                print(f"Error processing {fn}: {e}")

    print("All parallel tasks completed.")
    return results

def task_extract_data_and_generate(config_path: str, resume:bool, file_name: str):
    """
    2) Loads cleaned resumes from pickle, extracts structured data using LLM,
       and converts them into a list of summary sentences for each resume.
    Saves structured info + sentences to a pickle.
    """
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    if resume:
        existing_path = config["paths"]["candidatesProfile"]
        if os.path.exists(existing_path):
            already_existing_files = os.listdir(existing_path)

    output_folder = config["paths"]["pdf2text"]
    candidates_profile = config["paths"]["candidatesProfile"]
    llm_server_url = config["llm_server"]["base_url"]
    all_files = os.listdir(output_folder)
    if file_name is not None:
        all_files = [file_name]
    if resume:
        all_files = [file for file in all_files if file not in already_existing_files]
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
    processed_files = parallel_process_resumes(cleaned_resume_data,
                                               schema,
                                               llm_server_url,
                                               candidates_profile)
    print("Processed files:", processed_files)

def JSON_EXTRACT(config_path: str, file_name: str):
    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    candidates_profile = config["paths"]["candidatesProfile"]
    search_index = config["paths"]["search_index"]
    database_path = config["paths"]["database_path"]
    # Extract data + generate sentences
    structured_info = {}
    sentences_dict = {}
    database_lst = []
    all_files = os.listdir(candidates_profile)
    if file_name is not None:
        all_files = [file_name]

    failed_json = []
    for txt_file in all_files:
        if txt_file.endswith(".txt"):
            full_path = os.path.join(candidates_profile, txt_file)
            with open(full_path, "r") as file:
                text_data = file.read()
            # print(text_data)
            parsed = parse_resume_file_chatapi(text_data)
            if not parsed:
                logger.warning(f"No JSON found for {txt_file}")
                failed_json.append(full_path)
                continue
            try:
                structured_info[txt_file] = parsed
                temp_df = pd.DataFrame({key:[str(val)] for key,val in parsed.items()})
                temp_df["student_id"] = txt_file
                database_lst.append(temp_df)
                # Create a textual summary or "profile sentence" from parsed data
                # This is your "generate sentences for each resume" part
                summary_text = create_candidate_profile(parsed)
                sentences_dict[txt_file] = summary_text
            except Exception as e:
                logger.error(f"Parsing JSON for {txt_file} failed: {e}")

    results_pickle_path = os.path.join(search_index, "structured_resumes.pkl")
    with open(results_pickle_path, "wb") as f:
        pickle.dump({
            "structured_info": structured_info,
            "sentences_dict": sentences_dict
        }, f)
    logger.info(f"Structured data & sentences saved to: {results_pickle_path}")
    students_database = pd.concat(database_lst, axis=0)
    print("DATABASE PATH IS:", database_path)
    students_database.to_csv(f"{database_path}/students_database.csv", index=False)

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
    print("SIZE OF SENTENCES", len(sentences))
    print("SOME SAMPLE SENTENCES ARE", sentences[:5])
    embeddings = embedding_model.encode(sentences)
    print("SIZE OF EMBEDDINGS", embeddings.shape)

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
    print("STUDENTS EMBEDDINGS FILE PATH IS:", embeddings_pickle_path)
    if not os.path.isfile(embeddings_pickle_path):
        logger.error("No embeddings found. Run 'generate_embeddings' first.")
        return

    with open(embeddings_pickle_path, "rb") as f:
        data = pickle.load(f)

    print(data)
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

def task_get_insights(config_path: str):
    config = load_config(config_path)
    db_path = config["paths"]["database_path"]
    get_insights(db_path)



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
        choices=["split_pdf", "extract_data", "json_extract", "get_insights", "generate_embeddings", "index_embeddings", "run_api"],
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
    parser.add_argument(
        "--resume",
        default=False,
        help="Resume the extraction of data using models"
    )
    return parser.parse_args()

def main():
    print("inside main")
    setup_logger(logging.INFO)
    logger = logging.getLogger(__name__)

    args = parse_args()
    logger.info(f"Running function: {args.function}")
    logger.info(f"Resume argument is: {args.resume}")
    logger.info(f"File argument is: {args.file}")
    print(args)

    if args.function == "split_pdf":
        task_split_and_clean(args.config)
    elif args.function == "extract_data":
        task_extract_data_and_generate(args.config, args.resume, args.file)
    elif args.function =="json_extract":
        JSON_EXTRACT(args.config, args.file)
    elif args.function =="get_insights":
        task_get_insights(args.config)
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
