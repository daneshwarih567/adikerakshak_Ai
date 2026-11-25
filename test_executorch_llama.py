
from executorch.extension.llm.runner import (
    TextLLMRunner,
    GenerationConfig
)

MODEL_PATH = r"C:\Users\Daneshwari\et_workspace\executorch\models\Llama-3.2-1B-Instruct.pte"

TOKENIZER_PATH = r"C:\Users\Daneshwari\et_workspace\executorch\models\tokenizer.model"

print("Loading Llama 3.2 1B through ExecuTorch...")

runner = TextLLMRunner(
    MODEL_PATH,
    TOKENIZER_PATH
)

print("Model loaded successfully!")

prompt = (
    "You are AdikeRakshak AI, an agricultural assistant "
    "for arecanut farmers in Karnataka. "
    "Explain in simple English: "
    "What are the early symptoms of Mahali disease in arecanut?"
)

config = GenerationConfig(
    echo=False,
    max_new_tokens=100,
    temperature=0.2
)

generated_text = []

def collect_token(token):
    generated_text.append(token)
    print(token, end="", flush=True)

print("\nGenerating farmer advisory...\n")

runner.generate(
    prompt,
    config,
    token_callback=collect_token
)

print("\n\n===== FINAL LLAMA RESPONSE =====")
print("".join(generated_text))

print("\nEXECUTORCH LLAMA TEST FINISHED")
