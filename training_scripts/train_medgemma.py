from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. Load the Model (Gemma-2b-it is best for chat)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2b-it-bnb-4bit", # 4-bit uses 70% less VRAM
    max_seq_length = 2048,
    load_in_4bit = True,
)

# 2. Add the Specialist Layer (LoRA)
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank: increase to 32 if the AI isn't "smart" enough
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",
)

# 3. Load your D: drive JSON data
dataset = load_dataset("json", data_files="training_data.json", split="train")

# 4. Define the Training Settings
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "output", # Tell it to learn the 'output' style
    max_seq_length = 2048,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60, # Start small (60 steps) to see if it works
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
    ),
)

# 5. Start Training
trainer.train()

# 6. Save your new "MedGemma" Brain
model.save_pretrained("medgemma_lora_model")
tokenizer.save_pretrained("medgemma_lora_model")
print("MedGemma is trained and saved to D:/MedGemmaProject/medgemma_lora_model")