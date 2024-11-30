import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import psycopg2
import numpy as np

# Step 1: Load the dataset
data = pd.read_csv("updated_chatbot_data.csv", encoding='ISO-8859-1')

# Step 2: Load the embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L6-v2')

# Step 3: Generate embeddings for all questions at once
questions = data['question'].tolist()
embeddings = model.encode(questions)

# Save embeddings into the dataset
data['embedding'] = list(embeddings)

# Step 4: Apply clustering on embeddings
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
data['cluster'] = kmeans.fit_predict(embeddings)

# Save cluster centers to a file
np.save("cluster_centers.npy", kmeans.cluster_centers_)

# Step 5: Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="su_bot_data",
    user="postgres",
    port="5432",
    password="tiger"
)
cur = conn.cursor()

# Step 6: Create the table if it doesn't exist with additional constraints
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

# Step 7: Insert data into the database
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