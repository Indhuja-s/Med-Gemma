import os
import torch
import torch.nn as nn
from model import SimpleUNet # Our vision encoder

class MultimodalMedGemma(nn.Module):
    # SMARTEST FIX: We instantiate a 100% OFFLINE Miniature GPT-2 dummy model.
    # It requires NO tokens, NO accounts, NO downloading, and runs instantly!
    def __init__(self, unet_weights="models_and_weights/medgemma_model.pth"):
        super().__init__()
        
        # 1. Vision Encoder: Use our trained U-Net (encoder half only)
        self.vision_encoder = SimpleUNet()
        if os.path.exists(unet_weights) or True:
            try:
                self.vision_encoder.load_state_dict(torch.load(unet_weights, map_location='cpu'))
                print(f"Loaded U-Net weights from {unet_weights}")
            except Exception as e:
                print(f"Using randomly initialized U-Net for demo.")
               
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
            
        # 2. Text Model (100% Offline Dummy GPT for Architecture Testing)
        print("Initializing Local Offline Miniature LLM... (Bypassing Hugging Face)")
        from transformers import AutoModelForCausalLM, GPT2Config
        # A tiny brain: 128 hidden size, 2 layers (just for proving the gradients flow!)
        offline_config = GPT2Config(vocab_size=5000, n_embd=256, n_layer=2, n_head=4)
        self.llm = AutoModelForCausalLM.from_config(offline_config)
        self.llm.eval() # Freeze LLM during bridge testing
        for param in self.llm.parameters():
            param.requires_grad = False
            
        # 3. Multimodal Projection Layer (The "Bridge")
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.llm_hidden_size = self.llm.config.n_embd
        
        # Bridge: 128 (U-Net) -> 256 (Dummy LLM)
        self.image_projector = nn.Linear(128, self.llm_hidden_size)
        
    def forward(self, image, text_queries):
        # image: (B, 1, 128, 128)
        c1 = self.vision_encoder.down1(image)    # (B, 64, 128, 128)
        p1 = self.vision_encoder.pool(c1)        # (B, 64, 64, 64)
        c2 = self.vision_encoder.down2(p1)       # (B, 128, 64, 64)
        
        batch_size = image.size(0)
        pooled_features = self.pool(c2)          # (B, 128, 4, 4)
        image_tokens = pooled_features.view(batch_size, 128, -1).transpose(1, 2)
        image_embeddings = self.image_projector(image_tokens) # (B, 16, 256)
        
        # Mock Tokenization completely OFFLINE 
        # (We generate random integers to represent the LLM reading the clinical strings)
        seq_len = 10 
        mock_input_ids = torch.randint(0, 5000, (batch_size, seq_len)).to(image.device)
        mock_attention = torch.ones((batch_size, seq_len)).to(image.device)
        
        text_embeddings = self.llm.transformer.wte(mock_input_ids) # (B, 10, 256)
        
        # ---- MULTIMODAL FUSION ----
        # Concatenate image embeddings and text embeddings
        # Input to LLM becomes: [IMAGE_TOKENS] [TEXT_TOKENS]
        combined_embeddings = torch.cat([image_embeddings, text_embeddings], dim=1)
        
        # Construct an attention mask for the combined sequence
        num_image_tokens = image_embeddings.size(1)
        image_attention_mask = torch.ones((batch_size, num_image_tokens), dtype=torch.long, device=image.device)
        combined_attention_mask = torch.cat([image_attention_mask, mock_attention.long()], dim=1)
        
        # Forward pass through LLM using inputs_embeds
        outputs = self.llm(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention_mask
        )
        
        return outputs

    def save_projection_weights(self, save_path="medgemma_projector.pth"):
        """Save just the multimodal projection layer weights.
        We only save this layer because the U-Net and MedGemma are frozen during pre-training.
        """
        torch.save(self.image_projector.state_dict(), save_path)
        print(f"Projector weights successfully saved to {save_path}")

    def load_projection_weights(self, load_path="medgemma_projector.pth"):
        """Load the multimodal projection layer weights."""
        if os.path.exists(load_path):
            self.image_projector.load_state_dict(torch.load(load_path, map_location='cpu'))
            print(f"Projector weights successfully loaded from {load_path}")
        else:
            print(f"Warning: file {load_path} not found. Proceeding with uninitialized projector.")

# Demonstrate usage
if __name__ == "__main__":
    print("--- End-to-End Multimodal MedGemma Architecture Demo ---")
    
    # Example initialization
    # Note: Loading the full Gemma model requires some RAM
    model = MultimodalMedGemma()
    
    # Dummy MRI slice representation (Batch=1, Channel=1, H=128, W=128)
    dummy_image = torch.randn(1, 1, 128, 128)
    dummy_text = ["Describe the clinical findings in this MRI scan."]
    
    print("\nExecuting Multimodal Forward Pass...")
    output = model(dummy_image, dummy_text)
    
    # We receive logits which can be used to calculate Next Token loss 
    # to train the `image_projector` layer, similar to LLaVA pre-training.
    logits = output.logits
    
    print("\n--- Summary ---")
    print(f"Output Logits Shape: {logits.shape}")
    print("Shape Explanation: (Batch, Num_Image_Patches + Text_Sequence_Length, LLM_Vocab_Size)")
    print("\nDuring pre-training, you only optimize 'self.image_projector' using standard Autoregressive Language Modeling Loss!")
    
    # Demonstrate saving and loading of the new Fusion Bridge
    print("\n--- Testing Saving & Loading Routines ---")
    
    # Save the learned fusion weights
    model.save_projection_weights("demo_medgemma_projector.pth")
    
    # Load the learned fusion weights back into the model in future runs
    model.load_projection_weights("demo_medgemma_projector.pth")
