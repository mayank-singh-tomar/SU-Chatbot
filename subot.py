from flask import Flask, render_template, request, session, send_file
from io import BytesIO
from sentence_transformers import SentenceTransformer
import psycopg2
from groq import Client
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns


#  Groq client
client = Client(api_key="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")

# Initialize the Flask app
app = Flask(__name__, static_folder="static")
app.secret_key = "cf63d8e2ff10bac2c3cb65a96b4f1afe7f6da52f3561993a85ee2a88bc95ca34"  

# Preload the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

try:
    cluster_centers = np.load("cluster_centers.npy")
except FileNotFoundError:
    raise RuntimeError("cluster_centers.npy not found. Ensure the first script is executed successfully.")


# Initialize chat history
chat_history = []

def db_connection():
    return psycopg2.connect(
            host="localhost",
            database="su_bot_data",
            user="postgres",
            port="5432",
            password="tiger")

@app.route("/")
def login_page():
    return render_template('login_page.html')

@app.route("/home", methods=["POST"])
def home():
    mail_id = request.form.get("email")
    session['mail_id'] = mail_id  # Store mail ID in session
    chat_history.clear()
    return render_template('index.html', chat_history=chat_history, mail_id=mail_id)

def track_user_attempts(mail_id):
    MAX_ATTEMPTS = 10
    TIME_LIMIT = 60 * 60  # One hour in seconds
    now = time.time()

    attempts = session.get(f'attempts_{mail_id}', [])
    attempts = [timestamp for timestamp in attempts if now - timestamp < TIME_LIMIT]

    if len(attempts) >= MAX_ATTEMPTS:
        next_allowed_time = min(attempts) + TIME_LIMIT
        wait_time = int((next_allowed_time - now) / 60)
        return False, wait_time

    attempts.append(now)
    session[f'attempts_{mail_id}'] = attempts
    return True, None

@app.route("/ask", methods=["POST"])
def ask_question():
    global chat_history
    mail_id = session.get('mail_id')

    allowed, wait_time = track_user_attempts(mail_id)
    if not allowed:
        return render_template("index.html", chat_history=chat_history, mail_id=mail_id, time_limit_exceeded=True, wait_time=wait_time)

    user_query = request.form.get("question")
    if not user_query:
        return render_template("index.html", chat_history=chat_history, mail_id=mail_id, answer="Please enter a valid question.")

    chat_history.append({'type': 'user', 'text': user_query})

    try:
        # Generate the query embedding
        query_embedding = np.array(model.encode(user_query), dtype=np.float32)

     
        similarities = np.dot(cluster_centers, query_embedding)  # Cosine similarity approximation
        nearest_cluster = int(np.argmax(similarities))  # Convert numpy.int64 to Python int

       
        conn = db_connection()
        cur = conn.cursor()

        query = """
            SELECT question, answer
            FROM qa_embeddings
            WHERE cluster = %s
            ORDER BY embedding <-> %s::vector
            LIMIT 1;
        """
        cur.execute(query, (nearest_cluster, query_embedding.tolist()))  # Convert numpy array to list
        response = cur.fetchone()

        if response:
            context = response[1]
            prompt = f"{context}\nUser asked: {user_query}\nAnswer:"

            completion = client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=1,
                max_tokens=1024,
                top_p=1,
                stream=True,
            )

            answer = ""
            for chunk in completion:
                content = chunk.choices[0].delta.content
                if content:
                    answer += content
        else:
            answer = "Sorry, no relevant answer found."

        chat_history.append({'type': 'bot', 'text': answer})

    except Exception as e:
        answer = f"An error occurred: {str(e)}"
        chat_history.append({'type': 'bot', 'text': answer})

    finally:
        if 'cur' in locals() and not cur.closed:
            cur.close()
        if 'conn' in locals() and conn:
            conn.close()

    # Collect user data and store the cluster
    submit_user_data(mail_id, user_query, nearest_cluster)
    cluster=CLUSTER_FIELDS[nearest_cluster]

    return render_template("index.html", chat_history=chat_history, answer=answer, mail_id=mail_id, cluster=cluster)

def submit_user_data(mail_id, question_asked, cluster):
    conn = db_connection()
    cursor = conn.cursor()
    
   
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_data (
            mail_id TEXT,
            question_asked TEXT,
            cluster INTEGER
        );
    """)
    conn.commit()

    # Insert user data
    cursor.execute("""
        INSERT INTO user_data (mail_id, question_asked, cluster)  
        VALUES (%s, %s, %s);
    """, (mail_id, question_asked, cluster))
    conn.commit()

    cursor.close()
    conn.close()


@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    # Get feedback data from form
    question = request.form.get("question")
    response = request.form.get("response")
    rating = int(request.form.get("rating"))
    feedback = request.form.get("feedback", "")

    # Save feedback to database
    save_feedback(question, response, rating, feedback)

 
    return render_template("index.html", chat_history=chat_history, feedback_message="Thank you for your feedback!")


def save_feedback(question, response, rating, feedback):
    flagged = rating < 3
    conn = db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback (question, chatbot_response, user_rating, user_feedback, flagged_for_review)
        VALUES (%s, %s, %s, %s, %s)
    """, (question, response, rating, feedback, flagged))
    conn.commit()
    cursor.close()
    conn.close()

#user_graph

@app.route("/user_status")
def user_status():
   
    return render_template('user_status.html')


CLUSTER_FIELDS = {
    0: "Security/General Facilities",
    1: "Related to Sitare",
    2: "Internship/Placement",
    3: "Academics/Curriculum",
    4: "Campus/Facilities"
}

def fetch_data(user_id):
    connection = db_connection()
    cursor = connection.cursor()
    
    # Query to fetch cluster data for the specific user
    query = """
    SELECT cluster, COUNT(*) AS question_count
    FROM user_data
    WHERE mail_id = %s
    GROUP BY cluster
    ORDER BY cluster;
    """
    cursor.execute(query, (user_id,))
    data = cursor.fetchall()
    connection.close()
    
    # Convert to a dictionary for easier plotting
    cluster_data = {CLUSTER_FIELDS[row[0]]: row[1] for row in data}
    return cluster_data

@app.route("/user_status_graph", methods=["POST"])
def user_status_graph():
    # Get mail_id from the form
    user_id = request.form.get('mail_id')
    if not user_id:
        return "User ID not found.", 400

    cluster_data = fetch_data(user_id)
    if not cluster_data:
        return "No data available for this user.", 404

    # Generate the graph
    fields = list(cluster_data.keys())
    question_counts = list(cluster_data.values())
    
    try:
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        sns.barplot(x=fields, y=question_counts, palette="viridis")
        plt.xlabel("Fields", fontsize=12)
        plt.ylabel("Number of Questions Asked", fontsize=12)
        plt.title(f"Questions Asked by User ({user_id}) Per Field", fontsize=16)
        plt.xticks(rotation=45, ha="right", fontsize=10)
        plt.tight_layout()

        img = BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return send_file(img, mimetype='image/png')
    
    except Exception as e:
        return f"Error generating graph: {e}", 500


if __name__ == "__main__":
    app.run(debug=True)
