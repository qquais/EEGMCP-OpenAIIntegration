# EEGMCP-OpenAIIntegration

# EEG MCP Tool + OpenAI GPT Integration

## Overview

This project integrates BrainFlow-based EEG filtering with OpenAI’s GPT model to allow summarization and interpretation of EEG `.edf` files using natural language queries.

---

## 🔧 Requirements

- Python 3.10+
- Flask
- BrainFlow SDK
- NumPy
- OpenAI SDK (`openai>=1.0.0`)
- `.env` file with your `OPENAI_API_KEY`

---

## ✅ How It Works

### 1. EEG Filtering (`mcp_server.py`)
- Accepts `.edf` EEG files via `/filter-edf` POST endpoint.
- Applies a 0.5–40 Hz bandpass Butterworth filter to EEG channels using BrainFlow.
- Returns a JSON response with filtered signal values per channel.

### 2. GPT-Powered Summarization (`openai_agent.py`)
- Accepts `file` + `question` via `/agent/query`.
- If the question contains the word “filter”, it:
  - Calls `/filter-edf`
  - Computes summary stats (mean, min, max, std) from the filtered data
  - Sends stats + question to OpenAI's GPT-4 for intelligent summarization

---

## 🧪 Example Usage (Postman)

**POST** `http://localhost:5002/agent/query`

**Form Data:**
- `file`: `sample.edf`
- `question`: `Can you summarize the filtered EEG data?`

**Response:**
```json
{
  "status": "success",
  "answer": "The filtered EEG data consists of 16 channels..."
}
