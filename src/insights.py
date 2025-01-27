import pandas as pd
import ast
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def safe_eval(json_str):
    """
    Safely evaluate 'json-like' strings that might have single quotes
    or messy formatting. Return a parsed dictionary or an empty dict on failure.
    """
    try:
        return ast.literal_eval(json_str)
    except:
        return {}

def parse_skills(df):
    """
    Parse skills from the 'Skills' column and return skill frequency counters.
    """
    technical_skills = Counter()
    other_skills = Counter()

    for skills_str in df['Skills'].dropna():
        skills_dict = safe_eval(skills_str)
        tech = skills_dict.get('technical_skills', [])
        other = skills_dict.get('other_skills', [])

        technical_skills.update(skill.lower().strip() for skill in tech)
        other_skills.update(skill.lower().strip() for skill in other)

    return technical_skills, other_skills

def parse_education(df):
    """
    Extract CGPA/percentage values from the 'Education' column and count master's degrees.
    """
    all_cgpas = []
    masters_count = 0

    for edu_str in df['Education'].dropna():
        edu_dict = safe_eval(edu_str)

        # Count master's degrees
        if edu_dict.get('masters'):
            masters_count += 1

        # Extract CGPAs
        for key in ['pre_degree', 'bachelors', 'masters']:
            for record in edu_dict.get(key, []):
                cgpa = record.get('cgpa_or_percentage', None)
                try:
                    if cgpa:
                        all_cgpas.append(float(cgpa))
                except ValueError:
                    continue

    return all_cgpas, masters_count

def count_internships(df):
    """
    Count total internships across all students.
    """
    total_internships = 0

    for internship_str in df['Internships'].dropna():
        internship_list = safe_eval(internship_str)
        if isinstance(internship_list, list):
            total_internships += len(internship_list)

    return total_internships

def extract_skills(df):
    """
    Extract skills from the 'Skills' column, return a list of combined skills for each student.
    """
    skill_list = []
    for skills_str in df['Skills'].dropna():
        skills_dict = safe_eval(skills_str)
        tech_skills = skills_dict.get('technical_skills', [])
        other_skills = skills_dict.get('other_skills', [])
        combined_skills = tech_skills + other_skills
        skill_list.append(" ".join(combined_skills))  # Joining skills for vectorization
    return skill_list

def cluster_students(df, num_clusters=4):
    """
    Perform clustering on students based on extracted skills.
    """
    skill_texts = extract_skills(df)

    # Convert skill sets into a feature matrix using CountVectorizer
    vectorizer = CountVectorizer()
    skill_matrix = vectorizer.fit_transform(skill_texts)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(skill_matrix)

    # Display cluster sizes
    cluster_counts = Counter(df['Cluster'])
    print("\nCluster sizes:", cluster_counts)

    # Show cluster-wise sample students
    for cluster in range(num_clusters):
        print(f"\nCluster {cluster}:")
        print(df[df['Cluster'] == cluster][['Name', 'Skills']].head(5))

    return df, kmeans, vectorizer

def visualize_clusters(df):
    """
    Visualize the clusters using a bar chart.
    """
    cluster_counts = df['Cluster'].value_counts()
    plt.bar(cluster_counts.index, cluster_counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Students Across Clusters')
    plt.show()


# Analyze cluster representation
def analyze_clusters(df, vectorizer):
    feature_names = vectorizer.get_feature_names_out()
    cluster_counts = df['Cluster'].value_counts()

    cluster_summary = {}
    for cluster in df['Cluster'].unique():
        cluster_df = df[df['Cluster'] == cluster]
        all_skills = []
        for skills_str in cluster_df['Skills'].dropna():
            skills_dict = safe_eval(skills_str)
            all_skills.extend(skills_dict.get('technical_skills', []) + skills_dict.get('other_skills', []))

        skill_freq = Counter(all_skills)
        top_skills = skill_freq.most_common(5)
        cluster_summary[cluster] = top_skills

    return cluster_summary, cluster_counts

# Binning students based on the number of skills
def bin_students_by_skill_count(df):
    skill_counts = []
    for skills_str in df['Skills'].dropna():
        skills_dict = safe_eval(skills_str)
        skill_count = len(skills_dict.get('technical_skills', [])) + len(skills_dict.get('other_skills', []))
        skill_counts.append(skill_count)

    df['Skill Count'] = skill_counts
    bins = [0, 5, 10, 15, 20, max(skill_counts)]
    labels = ['0-5', '6-10', '11-15', '16-20', '20+']
    df['Skill Bin'] = pd.cut(df['Skill Count'], bins=bins, labels=labels, include_lowest=True)

    return df['Skill Bin'].value_counts()

# Visualization function
def visualize_skill_bins(skill_bin_distribution):
    skill_bin_distribution.plot(kind='bar', color='skyblue')
    plt.xlabel('Skill Count Bins')
    plt.ylabel('Number of Students')
    plt.title('Distribution of Students Based on Skill Count')
    plt.show()

def get_insights(file_path: str):
    # Replace with your actual CSV file path
    df = pd.read_csv(f"{file_path}/students_database.csv", encoding='utf-8')
    # 1) Parse and analyze skills
    tech_skills, other_skills = parse_skills(df)

    # 2) Education analysis
    # cgpa_list, masters_count = parse_education(df)

    # 3) Internship count
    total_internships = count_internships(df)

    # Generate insights
    total_students = len(df)
    # masters_ratio = (masters_count / total_students) * 100 if total_students > 0 else 0
    # avg_cgpa = sum(cgpa_list) / len(cgpa_list) if cgpa_list else 0

    # Display results
    print("=== Non-Trivial Insights from Student Resume Data ===\n")
    print(f"Total Students for which we were able to parse data correctly: {total_students}")
    # print(f"Students with a Masters degree: {masters_count} ({masters_ratio:.1f}%)")
    print(f"Total number of internships (summed across all students): {total_internships}")
    # print(f"\nAverage CGPA/Percentage (across all recorded degrees): {avg_cgpa:.2f}")

    print("\nTop 10 Technical Skills:")
    for skill, count in tech_skills.most_common(10):
        print(f"  {skill} => {count} mentions")

    print("\nTop 5 Other (Soft) Skills:")
    for skill, count in other_skills.most_common(5):
        print(f"  {skill} => {count} mentions")

    print("\n(You can expand this script to produce even more insights...)")
    # Run clustering
    df_clustered, model, vectorizer = cluster_students(df, num_clusters=4)
    cluster_summary, cluster_counts = analyze_clusters(df_clustered, vectorizer)
    skill_bin_distribution = bin_students_by_skill_count(df_clustered)

    # Visualize cluster distribution
    visualize_clusters(df_clustered)

    # Display insights
    print("\nCluster Summary (Top Skills in Each Cluster):")
    for cluster, skills in cluster_summary.items():
        print(f"Cluster {cluster}: {skills}")

    print("\nCluster Sizes:")
    print(cluster_counts)

    print("\nSkill Bin Distribution:")
    print(skill_bin_distribution)

    # Visualize skill bin distribution
    visualize_skill_bins(skill_bin_distribution)