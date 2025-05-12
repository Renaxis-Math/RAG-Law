from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from typing import Dict, Tuple
import json

ENTITY_CHECK_TEMPLATE = """
Analyze the following question and extract key entities related to:
1. Location (specifically if it mentions California)
2. Topic (specifically if it's related to insurance)

Question: {question}

You must respond with a valid JSON object in this exact format:
{
    "has_california": true/false,
    "has_insurance": true/false,
    "extracted_entities": {
        "location": ["list of location entities"],
        "topic": ["list of topic entities"]
    }
}
"""

# USED GPT FOR THIS
def check_entities(question: str, llm: ChatOpenAI) -> Tuple[bool, bool, str]:

    response = llm.invoke([
        SystemMessage(ENTITY_CHECK_TEMPLATE),
        HumanMessage(question)
    ])
    
    try:
        response_text = response.content
        
        # Find the first { and last } to extract the JSON object
        start = response_text.find('{')
        end = response_text.rfind('}') + 1
        if start == -1 or end == 0:
            raise ValueError("No JSON object found in response")
        
        json_str = response_text[start:end]
        response_data = json.loads(json_str)
        
        has_california = response_data["has_california"]
        has_insurance = response_data["has_insurance"]

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error parsing response: {e}")
        has_california = False
        has_insurance = False
    
    # Generate clarifying question if needed
    clarifying_question = ""
    if not has_california and not has_insurance:
        clarifying_question = "I notice your question doesn't mention California or insurance. Is your question related to California insurance law?"
    elif not has_california:
        clarifying_question = "I notice your question doesn't mention California. Are you asking about California insurance law specifically?"
    elif not has_insurance:
        clarifying_question = "I notice your question doesn't mention insurance. Are you asking about insurance-related matters in California?"
    
    return has_california, has_insurance, clarifying_question 
