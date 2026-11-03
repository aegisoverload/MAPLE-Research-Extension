
from hf_backend import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct")

params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100
)

prompts = [
    "Write a short scientific abstract about cancer research.",
    "Write a short scientific abstract about neuroscience."
]

outputs = llm.generate(prompts, params)

for i, output in enumerate(outputs):
    print(f"\n=== Output {i+1} ===")
    print(output.outputs[0].text)
