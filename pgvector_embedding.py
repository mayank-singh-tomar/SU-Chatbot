import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import psycopg2
import numpy as np

#  Load the dataset
data = pd.read_csv("updated_chatbot_data.csv", encoding='ISO-8859-1')

#clean_data
# def clean_text(text):
#     # Remove special characters and numbers
#     text = re.sub(r"[^a-zA-Z\s]", "", text)
#     # Convert text to lowercase
#     text = text.lower()
#     # Tokenize and remove stopwords
#     tokens = word_tokenize(text)
#     filtered_tokens = [word for word in tokens if word not in stop_words]
#     return " ".join(filtered_tokens)

# # Apply cleaning to questions and answers
# data['cleaned_question'] = data['question'].apply(clean_text)
# data['cleaned_answer'] = data['answer'].apply(clean_text)


model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

#  Generate embeddings for all questions at once
questions = data['question'].tolist()
embeddings = model.encode(questions)

# Save embeddings into the dataset
data['embedding'] = list(embeddings)

# Apply clustering on embeddings
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(embeddings)

# Save cluster centers to a file
np.save("cluster_centers.npy", kmeans.cluster_centers_)

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="su_bot_data",
    user="postgres",
    port="5432",
    password="tiger"
)
cur = conn.cursor()

# Create the table if it doesn't exist with additional constraints
cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS qa_embeddings (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL UNIQUE,  -- Ensure no duplicate questions
    answer TEXT NOT NULL,           -- Ensure the answer is provided
    cluster INT,
    embedding vector(384),
    CONSTRAINT unique_question UNIQUE (question)  -- Add a unique constraint on question
);
""")

#  Insert data into the database
insert_query = """
INSERT INTO qa_embeddings (question, answer, cluster, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (question) DO NOTHING;  -- Avoid inserting duplicates
"""
for _, row in data.iterrows():
    cur.execute(insert_query, (
        row['question'], 
        row['answer'], 
        row['cluster'], 
        row['embedding'].tolist()  # Convert numpy array to list
    ))

# Commit and close
conn.commit()
cur.close()
conn.close()

print("Data inserted and cluster centers saved.")
