import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, spearmanr
import seaborn as sns

skills = [
    "Python", "Java", "C++", "C#", "SQL", "PHP", "JavaScript", "HTML", "CSS", ".NET Framework", "ASP.NET",
    "MySQL", "Oracle 10g", "Database Management Systems (DBMS)", "PL/SQL",
    "Web Design", "Web Development", "REST API", "Drupal", "WordPress", "Ajax",
    "Visual Studio", "Eclipse", "Git", "MATLAB", "Android Studio",
    "Linux", "Windows Server", "Red Hat Linux", "Networking",
    "Adobe Photoshop", "Adobe Illustrator", "Adobe Premiere", "Adobe After Effects", "2D and 3D Graphics",
    "Artificial Intelligence", "Neural Networks", "Computer Vision", "Machine Learning", "Data Structures", "Digital Image Processing",
    "Operating Systems", "MATLAB", "Tableau"
]

skills = [x.lower() for x in skills]
skills_set = set([x.lower() for x in skills])

def filter_skills(row):
    return {
        "technical_skills": list(set(row.get("technical_skills", [])) - {"not found"}),
        "other_skills": list(set(row.get("other_skills", [])) - {"not found"}),
    }

def process_skills(row):
    return {
        "technical_skills": [s.lower() for s in row.get("technical_skills", []) if s.lower() in skills_set],
        "other_skills": [s.lower() for s in row.get("other_skills", []) if s.lower() in skills_set],
    }

def atleast_one_internship(row):
    temp = row
    flag=False
    for val in temp:
        try:
            if str(val["company"])!=str('not found'):
                flag=True
        except:
            pass
            
    return flag

def count_tech_skills(row):
    try:
        temp = row["technical_skills"]
    except:
        return 0
    return len(temp)

def count_other_skills(row):
    try:
        temp = row["other_skills"]
    except:
        return 0
    return len(temp)

def countProjects(row):
    count = 0
    for dct in row:
        try:
            if dct.get("project_name", False)=="not found":
                pass
            else:
                count+=1
        except:
            if dct!="not found" and len(dct)>=50:
                count+=1

    return count

def extract_grades(row):
    cgpa = -1
    if 'bachelors' in row.keys():
        try:
            cgpa = row['bachelors'][0]["cgpa_or_percentage"]
        except:
            cgpa = row['bachelors']["cgpa_or_percentage"]
    
    if str(cgpa) in ['',' ', 'not found', '0 present']:
        cgpa=-1
    elif len(str(cgpa).split(","))>1:
        cgpa = str(cgpa).split(",")[-1]
    elif float(cgpa)>10:
        cgpa=-1
    return cgpa


# 1. Correlation Analysis
def analyze_correlation(df):
    corr_matrix = df[['count_tech_skills', 'count_other_skills', 'cgpa']].corr(method='spearman')
    print("Correlation Matrix (Spearman):")
    print(corr_matrix)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

# 2. CGPA Distribution Based on Internship
def cgpa_distribution_analysis(df):
    sns.boxplot(x='is_internship', y='cgpa', data=df)
    plt.title("CGPA Distribution by Internship Status")
    plt.show()
    intern_cgpa = df[df['is_internship']]['cgpa']
    no_intern_cgpa = df[~df['is_internship']]['cgpa']
    t_stat, p_val = ttest_ind(intern_cgpa, no_intern_cgpa)
    print(f"T-test between CGPA of Internship vs No Internship: t-stat={t_stat:.2f}, p-value={p_val:.2e}")


# 4. Ratio of Tech to Other Skills
def skill_ratio_analysis(df):
    df['skill_ratio'] = np.where(
        (df['count_other_skills'] == 0), 
        np.inf, 
        df['count_tech_skills'] / df['count_other_skills']
    )
    sns.histplot(df['skill_ratio'], bins=10, kde=True, hue=df['is_internship'])
    plt.title("Skill Ratio Distribution")
    plt.xlabel("Tech Skills / Other Skills Ratio")
    plt.show()

# 5. Group-Level Insights
def group_insights(df):
    group_stats = df.groupby('is_internship').agg({
        'count_tech_skills': ['mean', 'median'],
        'count_other_skills': ['mean', 'median'],
        'cgpa': ['mean', 'median']
    })
    print("Group Statistics by Internship Status:")
    print(group_stats)

# 6. Relationship Between CGPA and Skills
def cgpa_skills_relationship(df):
    for col in ['count_tech_skills', 'count_other_skills']:
        corr, p_val = spearmanr(df[col], df['cgpa'])
        print(f"Spearman Correlation between CGPA and {col}: {corr:.2f}, p-value={p_val:.2e}")
        sns.scatterplot(x=col, y='cgpa', data=df, hue='is_internship')
        plt.title(f"Relationship Between CGPA and {col}")
        plt.show()

def basic_analysis(df):
    # Basic data exploration
    print("Data Overview:")
    print(df.describe())

    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    print("\nGender Distribution:")
    print(gender_counts)

    # Internship distribution
    internship_counts = df['is_internship'].value_counts()
    print("\nInternship Distribution:")
    print(internship_counts)



    # Average skill counts by internship status
    internship_skills = df.groupby('is_internship')[['count_tech_skills', 'count_other_skills', 'project_counts']].mean()
    print("\nAverage Skill Counts by Internship Status:")
    print(internship_skills)

    # Visualize
    internship_skills.plot(kind='bar', figsize=(10, 6), color=['orange', 'lightgreen', 'red'])
    plt.title('Tech and Other Skills by Internship Status')
    plt.ylabel('Average Skill Count')
    plt.xticks(rotation=0)
    plt.show()

    # Average skill counts by internship status
    internship_skills = df.groupby('is_internship')[['count_tech_skills', 'count_other_skills']].mean()
    print("\nAverage Skill Counts by Internship Status:")
    print(internship_skills)

    # Visualize
    internship_skills.plot(kind='bar', figsize=(10, 6), color=['orange', 'lightgreen'])
    plt.title('Tech and Other Skills by Internship Status')
    plt.ylabel('Average Skill Count')
    plt.xticks(rotation=0)
    plt.show()