
# Self-RAG Agent

A Self-RAG assistant built with Python, Gemini, ChromaDB, and LangGraph.

## Features

- Query routing: direct response, retrieval, or clarification
- Semantic retrieval from a sample handbook
- Passage relevance grading
- One query rewrite when retrieval finds no relevant passages
- Answer support evaluation and one regeneration attempt
- Automated tests with mocked model responses

## Requirements

- Python 3.13
- A Gemini API key

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` in the project root and add your key:

   ```dotenv
   GEMINI_API_KEY=your_actual_api_key
   ```

4. Run the assistant:

   ```bash
   python -m src.main
   ```

## Project structure

```
src/
  main.py     CLI entry point
  graph.py    LangGraph agent (route, retrieve, grade, rewrite, generate, critique)
  reflect.py  Gemini reflection steps (classify, grade, rewrite, critique)
  answer.py   Answer generation
  kb.py       ChromaDB knowledge base (created in data/chroma on first use)
  llm.py      Gemini client with rate limiting and retries
  config.py   Settings
tests/        Unit tests with mocked model responses
```

## Run tests

```bash
python -m pytest tests -v
```

## Security

Never commit `.env` or publish your Gemini API key.
The included handbook content is fictional and intended for demonstration.
