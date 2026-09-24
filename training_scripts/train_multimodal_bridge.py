import torch
import torch.nn as nn
import torch.optim as optim
from multimodal_medgemma_architecture import MultimodalMedGemma

def train_bridge():
    print("--- Multimodal Bridge (Projection Layer) Training Demo ---")
    print("Initializing Model (This may take a minute on CPU)...")
    
    # Load our End-to-End model
    # Note: This will load the frozen U-Net and Gemma models
    model = MultimodalMedGemma()
    
    # 1. Setup Optimizer
    # Notice we ONLY pass the parameters of the image_projector layer!
    # This ensures we don't accidentally train the LLM or U-Net and destroy their learned knowledge.
    optimizer = optim.Adam(model.image_projector.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    
    print("\nModel Initialized. All layers frozen except 'image_projector'.")
    print("Starting Dummy Training Loop...\n")
    
    model.train()
    
    # 2. Training Loop (Dummy Batch)
    epochs = 3
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Simulate a batch of 2 MRI slices
        dummy_images = torch.randn(2, 1, 128, 128) 
        
        # Simulate the text descriptions we want the AI to predict based on the images
        dummy_text = [
            "Clinical finding: 5cm tumor observed.", 
            "Clinical finding: Unremarkable scan."
        ]
        
        # Forward pass: Images and Text are fused
        outputs = model(dummy_images, dummy_text)
        logits = outputs.logits 
        
        # To compute loss for Next-Token Prediction in LLMs, we typically shift logits and labels.
        # For this demo, we'll create some dummy token targets matching the vocab size to prove the gradient flows back!
        batch_size, seq_len, vocab_size = logits.shape
        
        # Random target tokens for the sequence
        random_targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=logits.device)
        
        # Reshape for standard Cross Entropy Loss: (Batch * Seq_Len, Vocab_Size)
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = random_targets.view(-1)
        
        # Calculate loss
        loss = loss_fn(logits_flat, targets_flat)
        
        # Backward pass: Compute gradients
        loss.backward()
        
        # Update weights of ONLY the projection layer
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] | Bridge Loss: {loss.item():.4f}")
        
    print("\nTraining Complete! Saving the newly learned Bridge Weights...")
    
    # 3. Save only the bridge!
    model.save_projection_weights("medgemma_projector_learned.pth")
    print("You can now load this bridge alongside your U-Net and Gemma model for inference!")

if __name__ == "__main__":
    train_bridge()
