import asyncio
import logging
from llama_cpp import Llama
from config.config import DEEPSEEK_MODEL
from langchain_core.output_parsers import JsonOutputParser

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_llm_instance = None
_json_parser = JsonOutputParser()

def get_llm_instance():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    try:
        logger.info(f"✅ Loading DeepSeek LLM from: {DEEPSEEK_MODEL}")
        _llm_instance = Llama(
            model_path=str(DEEPSEEK_MODEL),
            n_ctx=128,
            n_threads=2,
            n_batch=1,
            n_gpu_layers=0,
            f16_kv=True,
            use_mmap=True,
            use_mlock=False,
            verbose=True
        )
        logger.info("✅ LLM loaded successfully")
        logger.info("🔍 Running LLM health check...")
        test_output = _llm_instance('{"reply": "pong"}', max_tokens=20, stop=["</s>"])
        reply_text = test_output['choices'][0]['text'].strip()
        logger.info(f"✅ Health Check LLM replied: {reply_text}")
    except Exception as e:
        logger.error(f"❌ Failed to load LLM: {e}", exc_info=True)
        _llm_instance = None
    return _llm_instance

async def warmup_llm():
    llm = get_llm_instance()
    if llm is None:
        logger.error("❌ LLM warmup failed: Model not loaded")
        return
    try:
        logger.info("🔧 Starting LLM warmup...")
        llm("Warmup prompt", max_tokens=10, stop=["</s>"], echo=False)
        logger.info("🔧 Warmup completed successfully")
    except Exception as e:
        logger.error(f"❌ Warmup failed: {e}", exc_info=True)

async def run_llm_prompt(prompts, parse_json=False):
    llm = get_llm_instance()
    if llm is None:
        logger.error("❌ LLM not initialized for inference.")
        return ['{"columns": {}}']
    responses = []
    for prompt in prompts:
        try:
            logger.info(f"➡️ Prompt: {prompt[:200]}...")
            result = llm(prompt, max_tokens=128, stop=["</s>"], echo=False)
            reply = result['choices'][0]['text'].strip()
            logger.info(f"⬅️ LLM Reply: {reply[:200]}...")
            if parse_json:
                logger.info("Parsing response as JSON")
                reply = _json_parser.parse(reply)
            responses.append(reply)
        except Exception as e:
            logger.error(f"❌ LLM inference failed for prompt: {e}", exc_info=True)
            responses.append('{"columns": {}}')
    return responses

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    llm = get_llm_instance()
    if llm:
        asyncio.run(warmup_llm())
        print("LLM is ready for inference.")
    else:
        print("LLM failed to load.")