# AI Resume Job Recommendation System

This project is an AI-based Resume Screening and Job Recommendation System that analyzes a user's CV and recommends suitable job roles based on job descriptions.

## Features
- Upload resume in PDF format
- Extracts text from resume automatically
- Uses NLP techniques to understand resume content
- Recommends top matching job roles
- Supports multiple job domains such as AI, Data Science, Software Engineering

## Technologies Used
- Python
- Flask
- Machine Learning
- Natural Language Processing (NLP)
- TF-IDF
- Cosine Similarity
- Pandas
- Scikit-learn

## Dataset
Due to Kaggle permission restrictions, the full dataset is kept private.
A sample dataset is included in this repository for demonstration.
Full dataset can be shared upon request.


## Project Structure
CV_job_recommender/
│
├── app.py
├── model/
│ └── jobs.csv
├── templates/
│ └── index.html
├── uploads/
├── README.md



## How to Run the Project

1. Clone the repository

git clone https://github.com/your-username/AI-Resume-Job-Recommender.git


2. Navigate to the project folder


cd CV_job_recommender


3. Install required libraries


python -m pip install flask pdfplumber pandas scikit-learn


4. Run the Flask app


python app.py


5. Open your browser and go to


http://127.0.0.1:5000/


## Output
The system displays the top recommended job roles along with similarity scores based on the uploaded resume.

## Future Improvements
- Add skill gap analysis
- Improve UI
- Deploy the application online
- Add deep learning based resume classification

## Author
Muhammad Hanzala