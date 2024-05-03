import google.generativeai as genai
from markdown_pdf import MarkdownPdf, Section
from google.colab import userdata
from datetime import datetime
import time
import argparse
import uuid
import pandas as pd
import os
import sys

# Set up the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 0,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
]

now = datetime.now()
actual_date = now.strftime("%Y-%m-%d - %H:%M:%S")

##############################################################
# Check file type
##############################################################

def check_file_type(file_name):
    """Check if the file is a csv, parquet, json, or avro file

    Parameters
    ----------
    file_name : str
        Name of the file to be converted

    Returns
    -------
    file_content : str
        Content of the file
    """
    # Check if the filex exist
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"File {file_name} not found")
    else:
        # Check the file type
        if file_name.endswith(".csv"):
            df = pd.read_csv(file_name)
        elif file_name.endswith(".parquet"):
            df = pd.read_parquet(file_name)
        elif file_name.endswith(".avro"):
            df = pd.read_avro(file_name)
        elif file_name.endswith(".json"):
            df = pd.read_json(file_name)
        else:
            raise ValueError(f"File type {file_name} not supported")
    
    df = df.dropna()  # delete row files with NaN values
    
    # Format float values to 2 decimal places:
    try:
        for col in df.select_dtypes(include='float64').columns:
            df[col] = df[col].apply(lambda x: "{:.2f}".format(x))
    except:
        print("Cannot format float values.")
    
    df = df.sample(250) # sample 250 rows from the dataframe
    df.to_csv("file_to_analyze.csv", sep=',', index=False)


##############################################################
# Generate PDF file
##############################################################

def generate_pdf(file_name, text_md):
    """Generate a PDF file from the text

    Parameters
    ----------
    text_md : str
        Text to be converted to PDF
    """
    # Create a new MarkdownPdf object
    pdf = MarkdownPdf(toc_level=2)
    
    # Create a new section
    pdf.add_section(Section(text_md, toc=False))
    
    # Set the title of the PDF
    pdf.meta["title"] = f"Analysis of {file_name}"
    
    id4 = uuid.uuid4()
    file_pdf = f"data-analysis-{id4}.pdf"
    # Save the PDF file
    pdf.save(file_pdf)
    
    return file_pdf


##############################################################
# Analyze the file using the Gemini model
##############################################################

def explain_file(file_name, google_api_key):
    """Explain the file using the Gemini model

    Parameters
    ----------
    google_api_key : str
        Google API key
    """
    
    genai.configure(api_key=google_api_key)
    
    with open('file_to_analyze.csv') as file:
        csv_data = file.read()
    
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings
    )
    
    prompt = "Explains this CSV file and all its columns. Indicates the potential uses of this data and which columns could cause problems. Do not show the results in any table."
    response_try = 3
    
    while response_try != 0:
        try:
            response = model.generate_content([prompt, csv_data])
            break
        except:
            response_try -= 1
            print("Error: The model is not working properly. Retrying the connection...")
            time.sleep(5)
    
    if response_try == 0:
        raise Exception("The model is not working properly. Try again later.")

    file_name_formatted = file_name.split("/")[0]
    text_md = f'<div style="text-align: right"> {actual_date} </div><br>'
    text_md += f'<div style="font-style: italic"> {file_name_formatted} </div>\n\n'
    text_md += '__________'
    text_md += '\n\n'
    text_md = text_md + " " + response.text.replace("CSV", "")
    
    prompt = "Discuss the pros and cons of different data visualization techniques for data analysis of this csv file in Python. Select only the ones you think are relevant."
    
    try:
        print("Generating data visualization techniques for data analysis of this file...")
        response = model.generate_content([prompt, csv_data])
        text_md = text_md + "\n" + response.text.replace("CSV", "")
    except:
        print("Error: The model is not working properly. A part of the report could not be created.")
        print("Missing part: Data visualization techniques for data analysis of this file.")
        
    prompt = "Explain how to optimize missing data, outliers, and duplicate data in pandas using best coding practices."
    
    try:
        print("Generating data optimization report in pandas...")
        response = model.generate_content([prompt, csv_data])
        text_md = text_md + "\n" + response.text.replace("CSV", "")
    except:
        print("Error: The model is not working properly. A part of the report could not be created.")
        print("Missing part: Optimize missing data analysis performance in pandas.")
    
    return text_md

##############################################################
# MAIN FUNCTION
##############################################################

def data_analyzer(file_name, google_api_key):
    """Main python function
    
    Parameters
    ----------
    file_name : str
        Name of the file to be converted
    google_api_key : str
        Gemini API key
    """
    
    # Check the file type
    print("Analyzing the format of the file...")
    check_file_type(file_name)
    print("File type analyzed successfully.")
    
    # Analyze the file
    print("Analyzing the data of the file. This may take a few minutes...")
    text_md = explain_file(file_name, google_api_key)
    print("File data analyzed successfully.")
    
    # Generate the report
    print("Generating the report...")
    file_report = generate_pdf(file_name, text_md)
    
    print("Report generated successfully.")
    print(f"Report {file_report} saved in the current directory.")
    
  


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze the data of a file with the help of AIa')
    parser.add_argument('--file_name', type=str, help='Name of the file to be converted', required=True)
    parser.add_argument('--api_key', type=str, help='Gemini API key', required=True)
    args = parser.parse_args()

    data_analyzer(args.file_name, args.google_api_key)
    
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              safety_settings=safety_settings
)