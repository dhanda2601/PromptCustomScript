# 📄 llm/llm_handler.py

import asyncio
import logging
from functools import partial
from typing import List, Optional
from pathlib import Path

from llama_cpp import Llama
from config.modelsettings import (
    DEEPSEEK_MODEL, LLM_CONTEXT_SIZE, LLM_BATCH_SIZE, LLM_NUM_THREADS,
    LLM_N_GPU_LAYERS, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    LLM_REPEAT_PENALTY, LLM_TOP_P
)

# Setup logging
logger = logging.getLogger(__name__)

# --- Load LLM --- #
def _initialize_llm() -> Optional[Llama]:
    model_path = Path(DEEPSEEK_MODEL)
    if not model_path.exists():
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    config = {
        "model_path": str(model_path),
        "n_ctx": LLM_CONTEXT_SIZE,
        "n_batch": LLM_BATCH_SIZE,
        "n_threads": LLM_NUM_THREADS,
        "n_gpu_layers": LLM_N_GPU_LAYERS,
        "f16_kv": True,
        "use_mlock": False,
        "seed": 42,
        "verbose": True,
    }

    try:
        logger.info(f"🔁 Loading LLM from: {model_path}")
        llm = Llama(**config)

        # Health check
        test = llm.create_chat_completion(
            messages=[{"role": "user", "content": "Ping"}],
            max_tokens=4,
            temperature=0
        )
        reply = test["choices"][0]["message"]["content"].strip()
        if not reply:
            raise RuntimeError("❌ Health check failed: No reply")
        logger.info(f"✅ LLM loaded. Ping reply: {reply}")
        return llm

    except Exception as e:
        logger.exception("🚨 Failed to initialize LLM")
        raise RuntimeError(f"LLM failed to load: {e}")

# --- Singleton accessor --- #
def get_llm_instance() -> Optional[Llama]:
    import streamlit as st
    if "llm_model" not in st.session_state or st.session_state.llm_model is None:
        try:
            st.session_state.llm_model = _initialize_llm()
        except Exception as e:
            logger.error(f"LLM init failed: {e}")
            return None
    return st.session_state.llm_model

# --- Async Inference --- #
async def run_llm_inference_async(prompt: str, temperature=None, max_tokens=None) -> str:
    try:
        llm = get_llm_instance()
        if not llm:
            raise RuntimeError("LLM not available")

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            partial(
                llm.create_chat_completion,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens or LLM_MAX_TOKENS,
                temperature=temperature or LLM_TEMPERATURE,
                stop=["</s>", "<|endoftext|>", "\n\n"],
                repeat_penalty=LLM_REPEAT_PENALTY,
                top_p=LLM_TOP_P
            )
        )
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"❌ Async inference failed: {e}")
        return ""

# --- Batch Async --- #
async def run_llm_inference_batch_async(prompts: List[str], max_concurrent=2) -> List[str]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def worker(prompt):
        async with semaphore:
            return await run_llm_inference_async(prompt)

    return await asyncio.gather(*(worker(p) for p in prompts))

# --- Warmup Ping --- #
async def warmup_llm():
    try:
        logger.info("🔥 Warming up LLM...")
        reply = await run_llm_inference_async("Say OK", max_tokens=2)
        logger.info(f"✅ Warmup reply: {reply}")
    except Exception as e:
        logger.warning(f"Warmup failed: {e}")
