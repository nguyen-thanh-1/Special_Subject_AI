from enum import Enum
from typing import Optional
from threading import Lock

from src.utils.logger import logger

class Mode(str, Enum):
    CHAT = "CHAT"
    INDEXING = "INDEXING"

class AppModeManager:
    def __init__(self):
        self._mode = Mode.CHAT
        self._lock = Lock()

    @property
    def current_mode(self) -> Mode:
        return self._mode

    def set_mode(self, new_mode: Mode) -> bool:
        with self._lock:
            if self._mode == new_mode:
                return True
            
            logger.info(f"[app_mode] Switching mode from {self._mode.value} to {new_mode.value}")
            self._mode = new_mode
            
            try:
                # Lazy import to avoid circular dependencies
                from src.utils.router import get_router
                from src.utils.embeddings import get_embedder
                from src.utils.llm import get_llm
                
                if self._mode == Mode.INDEXING:
                    # INDEXING MODE:
                    # 1. Unload Router and LLM from GPU to free VRAM
                    logger.info("[app_mode] Unloading Router and LLM...")
                    get_router().unload()
                    get_llm().unload()
                    
                    # 2. Move Embedding to GPU for faster indexing
                    logger.info("[app_mode] Loading Embedding to CUDA...")
                    get_embedder().set_device('cuda')
                    
                elif self._mode == Mode.CHAT:
                    # CHAT MODE:
                    # 1. Move Embedding to CPU
                    logger.info("[app_mode] Moving Embedding to CPU...")
                    get_embedder().set_device('cpu')
                    
                    # 2. Reload Router to GPU (LLM will lazy load when requested)
                    logger.info("[app_mode] Reloading Router...")
                    get_router().ensure_loaded()
                
                return True
            except Exception as e:
                logger.error(f"[app_mode] Failed to switch mode: {e}")
                return False

_app_mode_instance: Optional[AppModeManager] = None

def get_app_mode_manager() -> AppModeManager:
    global _app_mode_instance
    if _app_mode_instance is None:
        _app_mode_instance = AppModeManager()
    return _app_mode_instance
