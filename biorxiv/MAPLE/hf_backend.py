
import torch
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 128


@dataclass
class GeneratedText:
    text: str


@dataclass
class GenerationResult:
    outputs: list


class LLM:
    def __init__(self, model, max_model_len=None, **kwargs):
        print(f"Loading Transformers model: {model}")

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.model.eval()

    def generate(self, prompts, sampling_params):
        results = []

        for prompt in prompts:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(self.model.device)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=sampling_params.max_tokens,
                    do_sample=sampling_params.temperature > 0,
                    temperature=max(sampling_params.temperature, 1e-5),
                    top_p=sampling_params.top_p,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_ids = output_ids[0][inputs["input_ids"].shape[-1]:]

            text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )

            results.append(
                GenerationResult(outputs=[GeneratedText(text=text)])
            )

        return results
