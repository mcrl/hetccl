# run_mixtral.py

from model_mixtral import get_hf_mixtral
import torch

def main():
    model, tokenizer = get_hf_mixtral()
    model.eval()  # inference mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    prompt = "Once upon a time"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Generated:")
    print(generated_text)

if __name__ == "__main__":
    main()