from src.utils.llm import get_llm
from typing import List, Dict
import json

class FinancialAgent:
    def __init__(self):
        self.llm = get_llm()

    def process_chat(self, user_input: str, history: List[Dict[str, str]], system_prompt: str = None):
        """
        Coordinates the LLM to process a financial chat request.
        """
        # Logic for pre-processing or additional context could go here
        # For now, it simply delegates to the LLM
        return self.llm.generate_response(user_input, history, system_prompt=system_prompt)

    def rewrite_query(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """
        Rewrites a contextual query into a standalone query based on recent history.
        """
        if not history:
            return user_input.strip()
            
        # Keep the last 4 messages (2 turns) to prevent context overflow and save time
        recent_history = history[-4:]
        
        history_text = ""
        for msg in recent_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            history_text += f"{role}: {msg['content']}\n"
            
        prompt = (
            "You are a linguistic Query Rewriter AI. Your ONLY task is to rewrite the latest user query into a standalone query.\n"
            "You MUST resolve and replace any pronouns (nó, cái đó, họ, v.v.) in the latest query with the specific subjects they refer to from the conversation history.\n"
            "You MUST output the result in valid JSON format containing a single key 'rewritten_query'.\n"
            "DO NOT answer the user's query. DO NOT provide explanations.\n"
            "CRITICAL:\n"
            "1. Keep the query concise. Only replace the pronouns.\n"
            "2. If there are multiple subjects in the history, ALWAYS prioritize the MOST RECENT subject mentioned right before the query.\n\n"
            "--- Examples ---\n"
            "History:\n"
            "Người dùng: Vinamilk là gì?\n"
            "Trợ lý: Là công ty sữa lớn nhất Việt Nam.\n"
            "Query: Cổ phiếu của nó có tốt không?\n"
            "Output: {\"rewritten_query\": \"Cổ phiếu của công ty Vinamilk có tốt không?\"}\n\n"
            "History:\n"
            "Người dùng: Xin chào\n"
            "Trợ lý: Chào bạn, mình có thể giúp gì cho bạn?\n"
            "Query: Hôm nay bạn khỏe không?\n"
            "Output: {\"rewritten_query\": \"Hôm nay bạn khỏe không?\"}\n\n"
            "--- Your Task ---\n"
            "History:\n"
            f"{history_text}\n"
            f"Query: {user_input}\n"
            "Output:"
        )
        
        generator = self.llm.generate_response(prompt, history=[])
        rewritten = ""
        for chunk in generator:
            rewritten += chunk
            
        rewritten = rewritten.strip()
        
        # Cleanup potential model prefixes/markdown
        if rewritten.startswith("```json"):
            rewritten = rewritten[7:]
        elif rewritten.startswith("```"):
            rewritten = rewritten[3:]
        if rewritten.endswith("```"):
            rewritten = rewritten[:-3]
        rewritten = rewritten.strip()
        
        # Parse JSON
        try:
            parsed = json.loads(rewritten)
            rewritten_final = parsed.get("rewritten_query", "")
            if rewritten_final:
                rewritten = rewritten_final
        except Exception as e:
            from src.utils.logger import logger
            logger.warning(f"[rewrite] JSON parsing failed, using raw output: {e}")
            pass
            
        if not rewritten:
            return user_input.strip()
            
        from src.utils.logger import logger
        logger.info(f"[rewrite] Original: '{user_input}' -> Rewritten: '{rewritten}'")
        return rewritten

# Singleton helper
_agent_instance = None

def get_financial_agent():
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = FinancialAgent()
    return _agent_instance
