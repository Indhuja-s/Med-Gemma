import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the standard Gemma 2B model (Instruction Tuned)
# We use the 'float32' version because it's most compatible with CPUs
model_id = "google/gemma-2b-it" 

print("Loading MedGemma on your CPU... Please wait.")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cpu",      # This tells the AI to use your Intel processor
    torch_dtype=torch.float32
)

def ask_medgemma(instruction, mri_data):
    # Combine the instruction and your U-Net data
    prompt = f"Instruction: {instruction}\nInput: {mri_data}\nResponse:"
    
    # Prepare the text for the AI
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    print("\n--- MedGemma Analysis (Processing on CPU) ---")
    
    # Generate the response
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7
        )
    
    # Show the result
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response.split("Response:")[-1].strip())

# Test it with your U-Net findings
test_findings = "Patient: BraTS20_010. Tumor Volume: 4.8 cm3. Location: Frontal Lobe."
ask_medgemma("Provide a clinical summary based on these MRI findings.", test_findings)