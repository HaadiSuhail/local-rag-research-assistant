import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class HFLLM:
    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        load_in_4bit: bool = True,
        device: str = "cuda"
    ):
        print(f"🔹 Loading LLM: {model_name}")

        self.device = device

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        else:
            bnb_config = None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.float16
        )

        self.model.eval()

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 150,
        repetition_penalty: float = 1.1
    ) -> str:

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                do_sample=False
            )

        generated_text = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        return generated_text