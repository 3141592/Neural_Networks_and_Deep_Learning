import subprocess
import time
import requests
from langchain_community.llms import Ollama
from langchain.evaluation import load_evaluator

def start_ollama_if_needed(model="mistral"):
    # Try a test request to see if Ollama is running and the model is ready
    try:
        test = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": "Hello",
            "stream": False
        })
        if test.status_code == 200:
            print(f"Ollama model '{model}' is already running.")
            return
    except requests.exceptions.ConnectionError:
        pass

    # Start Ollama model
    print(f"Starting Ollama model: {model}")
    subprocess.Popen(["ollama", "run", model])
    time.sleep(5)  # Wait a few seconds for it to be ready

# --- Step 1: Start Ollama if needed ---
model_name = "mistral"
start_ollama_if_needed(model_name)

# --- Step 2: Set up LangChain evaluator using Ollama ---
llm = Ollama(model=model_name)

evaluator = load_evaluator("labeled_criteria", llm=llm, criteria="helpfulness")

result = evaluator.evaluate_strings(
    input="What is the capital of France?",
    prediction="The capital of France is Paris.",
    reference="Paris"
)

print(result)

