import torch

def medgemma_reasoning(vision_stats):
    # This represents the "Brain" of the LLM receiving your U-Net stats
    tumor_vol = vision_stats['volume']
    
    # Logic for the LLM to decide on severity
    if tumor_vol > 5.0:
        severity = "High: Immediate surgical consultation recommended."
    elif tumor_vol > 1.0:
        severity = "Moderate: Follow-up contrast MRI in 4 weeks."
    else:
        severity = "Low: Monitoring protocol."

    response = f"""
    [MedGemma Analysis]
    I have reviewed the segmentation output from the U-Net.
    The detected volume is {tumor_vol} cm³. 
    
    Clinical Assessment: {severity}
    Potential Diagnosis: Likely Glioma.
    
    Would you like me to generate a referral letter for Neurology?
    """
    return response

# Simulated input from your previous vision scripts
stats = {'volume': 4.56, 'id': 'BraTS20_010'}
print(medgemma_reasoning(stats))