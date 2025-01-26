# Import Libraries 
import pandas as pd

# UDF function to load the data for resumes and job descriptions
def load_data(resumes_path, job_desc_path):
    """
    Load resume and job description data.
    Inputs: resumes_path (str): Path to resumes CSV file
            job_desc_path (str): Path to job descriptions CSV file
    Ouputs: tuple: Resumes and job descriptions dataframes
    """
    resumes = pd.read_csv(resumes_path, delimiter=";")  # Change the delimiter if needed
    job_descriptions = pd.read_csv(job_desc_path, delimiter=";")  # Change the delimiter if needed
    return resumes, job_descriptions
