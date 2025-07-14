from peft import PeftModel

from transformers import AutoModelForCausalLM, AutoTokenizer

import torch



model_name = "cyberagent/open-calm-1b"

adapter_path = "./adapter"

torch_dtype=torch.float16 

# Load the base model

base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)



# Load the tokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:

    tokenizer.pad_token = tokenizer.eos_token



# Load the adapter onto the base model

model = PeftModel.from_pretrained(base_model, adapter_path)



print("Model with fp16 adapter loaded successfully.")

print(f"Model device: {model.device}")

print(f"Tokenizer pad token: {tokenizer.pad_token}")


prompts_for_inference = [

    f"固有ID_{prefix}K{i}の情報を展開せよ"

    for prefix in range(10, 80, 10)  # 10, 20, 30, 40, 50

    for i in range(1, 11)           # 1, 2, ..., 10

]



for i, prompt in enumerate(prompts_for_inference):

    print(f"--- Inference for Prompt {i+1}: {prompt} ---")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():

        outputs = model.generate(

            **inputs,

            max_new_tokens=1200,

            do_sample=False,

            pad_token_id=tokenizer.eos_token_id # Use eos_token_id for padding

        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(generated_text)

    print("-" * 20)



print("Inference complete for all prompts.")