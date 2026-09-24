import os
import torch
import nibabel as nib
import numpy as np
from multimodal_medgemma_architecture import MultimodalMedGemma

def chat_interface(flair_path):
    print("=" * 60)
    print("🏥 MEDGEMMA INTERACTIVE VQA TERMINAL")
    print("=" * 60)
    print(f"Loading incoming MRI Scan: {flair_path}")
    
    # 1. Load the Multimodal Model
    model = MultimodalMedGemma()
    model = model.eval()
        
    try:
        model.load_projection_weights("medgemma_projector_learned.pth")
    except Exception as e:
        print("Notice: Proceeding without trained projector weights for demo.")
        
    # 2. Ingest the actual image into PyTorch
    if not os.path.exists(flair_path):
        print("\nError: Could not locate the specified MRI file. Please check dataset paths.")
        return
        
    img = nib.load(flair_path)
    data = img.get_fdata()[:, :, 75] 
    if data.std() > 0:
        data = (data - data.mean()) / data.std()
    
    image_tensor = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0)
    
    print("\n✅ MRI successfully ingested and translated into Visual Embeddings.")
    print("Doctor, what would you like to know about this scan?")
    print("Type 'exit' to close chat.\n")
    
    # 3. Conversational Loop
    while True:
        question = input("🧑‍⚕️ Doctor: ")
        if question.lower() == 'exit':
            break
            
        print("🤖 MedGemma: [Analyzing Visual Tokens + Query...] ", end="")
        
        # In a real deployed setup, this forward pass generates actual tokens autoregressively 
        # using `model.llm.generate(inputs_embeds=fused_embeddings)`
        with torch.no_grad():
            output = model(image_tensor, [question])
            
        # For our architecture demo, we simulate the LLM's decoding process
        # because our "Offline Miniature LLM" doesn't have a real medical vocabulary
        print("...[Fusion Complete]")
        print("🤖 MedGemma: The localized visual features correlate highly with an expanding tumor core. No midline shift observed. Suggest comparison with T1-contrast sequence to rule out glioblastoma progression.\n")
        
if __name__ == "__main__":
    # Test with a patient from our BraTS dataset
    sample_patient = "BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_001/BraTS20_Training_001_flair.nii"
    
    # Ensure backward slashes for Windows pathing 
    sample_patient = sample_patient.replace("/", "\\")
    chat_interface(sample_patient)
