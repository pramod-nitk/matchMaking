import pandas as pd
import ast
import os
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, spearmanr

from utils.insight_utils import (
    filter_skills,
    process_skills,
    atleast_one_internship,
    count_other_skills,
    count_tech_skills,
    countProjects,
    extract_grades,
    analyze_correlation,
    cgpa_distribution_analysis,
    group_insights,
    basic_analysis
)


        

def get_insights(file_path: str):
    # Replace with your actual CSV file path
    if os.path.exists(file_path):
        df = pd.read_csv(f"{file_path}/students_database.csv")
    else:
        raise FileNotFoundError("Data file not found.")
    
    columns_to_parse = ["Skills", "Projects", "Education", "Internships", "References"]

    for col in columns_to_parse:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x.lower()))

    columns_to_standardize = ["Name", "DateOfBirth", "Gender", "Email"]

    for col in columns_to_standardize:
        df[col] = df[col].apply(lambda x: x.lower())

    df["filtered_skills"] = df["Skills"].apply(filter_skills)
    df["filtered_skills"] = df["filtered_skills"].apply(process_skills)
    df["is_internship"] = df["Internships"].apply(atleast_one_internship)
    df["count_tech_skills"] = df["Skills"].apply(count_tech_skills)
    df["count_other_skills"] = df["Skills"].apply(count_other_skills)

    df["project_counts"] = df["Projects"].apply(countProjects)
    df["cgpa"] = df["Education"].apply(extract_grades)
    df["cgpa"] = df["cgpa"].apply(lambda x: float(str(x).strip()))

    basic_analysis(df)
    analyze_correlation(df)
    cgpa_distribution_analysis(df)
    group_insights(df)

            