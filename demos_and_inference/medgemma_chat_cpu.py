import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Using the official Google Gemma 2B (Instruction Tuned)
model_id = "google/gemma-2b-it" 

print("Step 1: Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Step 2: Loading Model into System RAM (this takes about 1-2 minutes)...")
# 'device_map="cpu"' is the most important part here!
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

def ask_medgemma(instruction, mri_data):
    # This template helps the AI understand it's a doctor's assistant
    prompt = f"System: You are a medical AI assistant. Analyze the following data.\nInstruction: {instruction}\nInput: {mri_data}\nResponse:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    print("\n--- MedGemma is analyzing (CPU calculation in progress) ---")
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    end_time = time.time()
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the output to show only the AI's response
    final_text = response.split("Response:")[-1].strip()
    print(f"\nRESULT:\n{final_text}")
    print(f"\n(Analysis took {end_time - start_time:.2f} seconds)")

# Test with your project data
test_input = "Patient ID: BraTS20_010. Findings: Tumor detected in Flair sequence. Volume: 4.85 cm3. Location: Frontal Lobe."
ask_medgemma("Provide a professional radiologist summary and next steps.", test_input)