from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import requests
import json

class LocalApi(LLM):
        
    @property
    def _llm_type(self) -> str:
        return "custom"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        print("Prompt: " + prompt)
        url = "http://86.27.34.44"
        payload={'message': prompt}
        headers = {
            'Content-Type': 'application/json; charset=UTF-8'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        return response.text
