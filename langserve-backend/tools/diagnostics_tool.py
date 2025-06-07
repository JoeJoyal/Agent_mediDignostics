from langchain.tools import tool 
from utils.euri_client import euri_chat_completion


@tool
def ai_diagnos(symptom_descriptoin):
    """use euri to provide the diagnosis suggestion based on the symptom given by users"""
    
    message = [{"role":"user","content":f"a patient reports : {symptom_descriptoin} .what are the possible diagnoses and next step and even suggest me cure for this "}]
    
    return euri_chat_completion(messages=message)