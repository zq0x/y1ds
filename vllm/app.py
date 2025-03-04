from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from datetime import datetime
import os
import torch
import pynvml
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize NVML once (e.g., at application startup)
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        logger.info('[START] [initialize_nvml] NVML initialized successfully.')
    except Exception as e:
        logger.error(f'[START] [initialize_nvml] Failed to initialize NVML: {e}')

initialize_nvml()

# Shutdown NVML once (e.g., at application exit)
def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
        logger.info('[START] [shutdown_nvml] NVML shutdown successfully.')
    except Exception as e:
        logger.error(f'[START] [shutdown_nvml] Failed to shutdown NVML: {e}')

# Check if CUDA is available
def cuda_support_bool():
    try:
        res_cuda_support = torch.cuda.is_available()
        if res_cuda_support:
            logger.info('[START] [cuda_support_bool] CUDA found')
        else:
            logger.error('[START] [cuda_support_bool] CUDA not supported!')
        return res_cuda_support
    except Exception as e:
        logger.error(f'[START] [cuda_support_bool] {e}')
        return False

# Return the number of CUDA GPUs available
def cuda_device_count():
    try:
        res_gpu_int = torch.cuda.device_count()
        res_gpu_int_arr = [i for i in range(res_gpu_int)]
        logger.info(f'[START] [cuda_support_bool] {str(res_gpu_int)}x GPUs found {res_gpu_int_arr}')
        return res_gpu_int_arr
    except Exception as e:
        logger.error(f'[START] [cuda_device_count] Failed to get device_count. Using default [0]. Error: {e}')
        return [0]

gpu_int_arr = cuda_device_count()

# FastAPI app
app = FastAPI()

# Global LLM instance
llm_instance = None

@app.get("/")
async def root():
    return f'Hello from server {os.getenv("CONTAINER_PORT")}!'

@app.post("/vllmt")
async def vllmt(request: Request):
    global llm_instance
    try:
        req_data = await request.json()
        logger.info(f'req_data: {req_data}')

        if req_data["req_type"] == "load":
            try:
                model_name = req_data["req_model"]
                logger.info(f'Loading model: {model_name}...')

                free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Free memory in GB
                logger.info(f'Available GPU memory: {free_memory:.2f} GB')

                model_pretrained = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto")
                model_size = sum(p.numel() * p.element_size() for p in model_pretrained.parameters()) / (1024 ** 3)  # Size in GB
                logger.info(f'Model size: {model_size:.2f} GB')

                llm_instance = LLM(
                    model=model_name,
                    tensor_parallel_size=1,
                    gpu_memory_utilization=0.32,
                    trust_remote_code=True
                )
                logger.info('Model loaded successfully.')

                return JSONResponse({"result_status": 200, "result_data": "Model loaded successfully"})

            except Exception as e:
                logger.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})

        elif req_data["req_type"] == "generate":
            try:
                if llm_instance is None:
                    raise HTTPException(status_code=400, detail="No model loaded. Please load a model first.")

                prompt = req_data.get("req_prompt", "The future of AI is")
                temperature = req_data.get("req_temperature", 0.8)
                top_p = req_data.get("req_top_p", 0.95)
                max_tokens = req_data.get("req_max_tokens", 100)

                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )

                logger.info(f'Generating text for prompt: {prompt}')
                outputs = llm_instance.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text

                return JSONResponse({"result_status": 200, "result_data": generated_text})

            except Exception as e:
                logger.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 500, "result_data": str(e)})

        else:
            raise HTTPException(status_code=400, detail="Invalid request type.")

    except Exception as e:
        logger.error(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))
    





# if req_data["req_model"] not in llm_instances:
#     print(f'MODEL NOT IN STANCES EGAAAAAAAL')
# llm_instances[req_data["req_model"]] = LLM(model=req_data["req_model"], task=req_data["req_task"])

# print("llm_instances")
# print(llm_instances)
# sampling_params = SamplingParams(int(temperature=req_data["req_temperature"]), top_p=0.9, max_tokens=100)

# outputs = llm_instances[req_data["req_model"]].generate([req_data["req_prompt"]], sampling_params)
# return JSONResponse({"result_status": 200, "result_data": outputs[0].outputs[0].text})
