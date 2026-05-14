import os, sys, json
from pathlib import Path

if __package__ in {None, ""}:
	sys.path.append(str(Path(__file__).resolve().parents[2]))

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from dotenv import load_dotenv

class apikeyManager:
    REQUIRED_KEYS = ["GROQ_API_KEY", "GOOGLE_API_KEY"]
    
    def __init__(self):
        self.api_keys = {}
        
        for key in self.REQUIRED_KEYS:
            if not self.api_keys.get(key):
                env_val = os.getenv(key)
                if env_val:
                    self.api_keys[key] = env_val
                    log.info(f"Loaded {key} from .env file")
                    
        missing = [k for k in self.REQUIRED_KEYS if not self.api_keys.get(k)]
        
        if missing:
            log.error(f"Missing api keys: {missing}")
            raise DocumentPortalException("Missing Required API Keys", sys)
        
    def load_key(self, key:str) -> str:
        key_val = self.api_keys.get(key)
        
        if not key_val:
            raise KeyError(f"API Key for {key} is missing")
        return key_val
        
    
class ModelLoader:
    """
    Loads embedding model and LLMS based on config and environment
    """
    
    def __init__(self):
        if os.getenv("ENV", "local") != "production":
            load_dotenv()
            log.info("Running in local mode, .env loaded")
        else:
            log.info("Running Production mode")
            
        self.api_key_mgr = apikeyManager()
        self.config = load_config()
        log.info(f"YAML config loaded {list(self.config.keys())}")
            
    def load_embeddings(self):
        try:
            model_name = self.config["embedding_model"]["model_name"]
            log.info("Loading embedding model", model=model_name)
            return GoogleGenerativeAIEmbeddings(model=model_name,
                                                google_api_key=self.api_key_mgr.api_keys.get("GOOGLE_API_KEY"))    
        except Exception as e:
            log.error("error during loading embedding model", error=str(e))
            raise DocumentPortalException("Failed to load embedding model", sys)
        
    def load_llm(self):
        llm_block = self.config["llm"]
        provider = os.getenv("PROVIDER","google")
        
        if provider not in llm_block:
            log.error("LLM Provider not found in config", provider=provider)
            raise ValueError(f"LLM provider: {provider} not found in config")
        
        llm_config = llm_block[provider]
        
        llm_provider = llm_config.get("provider")
        model_name = llm_config.get("model_name")
        temperature = llm_config.get("temperature", .2)
        max_output_tokens = llm_config.get("max_output_tokens", 2048)
        
        if llm_provider == "google":
            return ChatGoogleGenerativeAI(
                model=model_name,
                api_key = self.api_key_mgr.api_keys.get("GOOGLE_API_KEY"),
                temperature = temperature,
                max_tokens = max_output_tokens
            )
        elif llm_provider == "groq":
            return ChatGroq(
                model=model_name,
                api_key=self.api_key_mgr.api_keys.get("GOOGLE_API_KEY"),
                temperature=temperature
            )
        else:
            log.error("Unsupported llm provider ", provider=llm_provider)
            raise ValueError(f"Unsupported llm provider: {llm_provider}")
        
if __name__ == "__main__":
    loader = ModelLoader()
    
    embeddings = loader.load_embeddings()
    result = embeddings.embed_query("hi")
    print(f"\nembedding dim is : {len(result)}")
    
    model = loader.load_llm()
    print(f"\nLoaded llm: {model}\n")
    response = model.invoke("Hi! How are you?")
    print(f"\nllm response: \n{response.content}") 
        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            