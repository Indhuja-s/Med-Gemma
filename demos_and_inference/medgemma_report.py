import pandas as pd

def generate_llm_report(patient_id, volume_mm3, slice_count):
    # Convert mm3 to cm3 for medical standard
    volume_cm3 = volume_mm3 / 1000
    
    # Define the "Clinical Context"
    status = "Elevated" if volume_cm3 > 2.0 else "Minor"
    
    # This is the prompt we would send to Gemma
    prompt = f"""
    SYSTEM: You are a neuroradiologist assistant. 
    INPUT DATA:
    - Patient: {patient_id}
    - Modality: MRI FLAIR
    - Detected Tumor Volume: {volume_cm3:.2f} cm³
    - Number of slices affected: {slice_count}
    
    TASK: Write a formal medical finding.
    """
    
    # For now, let's simulate the high-quality Gemma output:
    report = f"""
    --- MEDGEMMA RADIOLOGY FINDINGS ---
    PATIENT ID: {patient_id}
    
    FINDINGS:
    Volumetric analysis of the T2-FLAIR sequence demonstrates a localized 
    hyperintense region consistent with a space-occupying lesion. 
    The estimated tumor volume is {volume_cm3:.2f} cm³. 
    The mass effect appears {status}.
    
    IMPRESSION:
    Findings are highly suggestive of glioblastoma activity. 
    Clinical correlation and contrast-enhanced T1 imaging are recommended for further characterization.
    """
    return report

# Let's use the data from our last run
print(generate_llm_report("BraTS20_Training_010", 4560.2, 15))