# backend/openai_agent.py
from flask import Flask, request, jsonify
import requests
import os
import openai
import numpy as np
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize OpenAI client (new SDK >= 1.0.0)
client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)

@app.route("/agent/query", methods=["POST"])
def agent_query():
    try:
        file = request.files.get("file")
        question = request.form.get("question")

        if not question:
            return jsonify({"error": "Question is required"}), 400

        # ✅ Tool-use: handle EEG filtering
        if file and "filter" in question.lower():
            response = requests.post("http://localhost:5000/filter-edf", files={"file": file})
            if response.status_code != 200:
                return jsonify({"error": "Failed to get EEG data"}), 500

            eeg_data = response.json().get("filtered_data", {})

            # ✅ Summarize EEG signal stats to reduce token count
            summary = {}
            for ch, values in eeg_data.items():
                arr = values[:1000]  # Only first 1000 points to limit size
                summary[ch] = {
                    "mean": round(np.mean(arr), 4),
                    "min": round(np.min(arr), 4),
                    "max": round(np.max(arr), 4),
                    "std": round(np.std(arr), 4)
                }

            prompt = f"Question: {question}\n\nEEG Channel Summary Stats:\n{summary}\n\nAnswer:"
        else:
            prompt = question

        # ✅ OpenAI API call
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        answer = completion.choices[0].message.content
        return jsonify({"status": "success", "answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5002)