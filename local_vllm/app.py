from fastapi import FastAPI, Request, HTTPException
import json
import subprocess
import docker
from docker.types import DeviceRequest
import time
import os
import redis.asyncio as redis
import sys
from fastapi.responses import JSONResponse
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
import pynvml
from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM
import tiktoken

r = redis.Redis(host="redis", port=6379, db=0)

# Initialize NVML once (e.g., at application startup)
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        print('[START] [initialize_nvml] NVML initialized successfully.')
    except Exception as e:
        print(f'[START] [initialize_nvml] Failed to initialize NVML: {e}')

initialize_nvml()

# Shutdown NVML once (e.g., at application exit)
def shutdown_nvml():
    try:
        pynvml.nvmlShutdown()
        print('[START] [shutdown_nvml] NVML shutdown successfully.')
    except Exception as e:
        print(f'[START] [shutdown_nvml] Failed to shutdown NVML: {e}')

# Check if CUDA is available
def cuda_support_bool():
    try:
        res_cuda_support = torch.cuda.is_available()
        if res_cuda_support:
            print(f'[START] [cuda_support_bool] CUDA found')
        else:
            print(f'[START] [cuda_support_bool] CUDA not supported!')
        return res_cuda_support
    except Exception as e:
        print(f'[START] [cuda_support_bool] {e}')
        return False

# Return the number of CUDA GPUs available
def cuda_device_count():
    try:
        res_gpu_int = torch.cuda.device_count()
        res_gpu_int_arr = [i for i in range(res_gpu_int)]
        print(f'[START] [cuda_support_bool] {str(res_gpu_int)}x GPUs found {res_gpu_int_arr}')
        return res_gpu_int_arr
    except Exception as e:
        print(f'[START] [cuda_device_count] Failed to get device_count. Using default [0]. Error: {e}')
        return [0]

gpu_int_arr = cuda_device_count()




llm = None

def get_gpu_info():
    try:
        
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_util = utilization.gpu
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        mem_used = memory_info.used / 1024**2
        mem_total = memory_info.total / 1024**2
        mem_util = (mem_used / mem_total) * 100

        gpu_info.append({
                "gpu_count": int(device_count),
                "gpu_util": float(gpu_util),
                "mem_used": float(mem_used),
                "mem_total": float(mem_total),
                "mem_util": float(mem_util)
        })
        pynvml.nvmlInit()
        pynvml.nvmlShutdown()     
        return gpu_info
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

current_gpu_info = get_gpu_info()
print("current_gpu_info")
print(current_gpu_info)
print("current_gpu_info[0]['gpu_count']")
print(current_gpu_info[0]["gpu_count"])
gpu_int_arr = [i for i in range(current_gpu_info[0]["gpu_count"])]
print("gpu_int_arr")
print(gpu_int_arr)
async def redis_timer():
    while True:
        try:
            for gpu_i in gpu_int_arr:
                current_gpu_info = get_gpu_info()
                res_db_gpu = await r.get('db_gpu')
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)
                    updated_gpu_data = []
                    for gpu_int in range(len(db_gpu)):

                        update_data = {
                            "gpu": str(gpu_i),
                            "gpu_info": str(current_gpu_info),
                            "running_model": db_gpu[gpu_int].get("running_model", "0"),
                            "timestamp": str(datetime.now()),
                            "port_vllm": db_gpu[gpu_int].get("port_vllm", "0"),
                            "port_model": db_gpu[gpu_int].get("port_model", "0"),
                            "used_ports": db_gpu[gpu_int].get("used_ports", "0"),
                            "used_models": db_gpu[gpu_int].get("used_models", "0"),
                        }
                        updated_gpu_data.append(update_data)
                    await r.set('db_gpu', json.dumps(updated_gpu_data))
                else:
                    update_data = [{
                        "gpu": 0,
                        "gpu_info": str(current_gpu_info),
                        "running_model": "0",
                        "timestamp": str(datetime.now()),
                        "port_vllm": "0",
                        "port_model": "0",
                        "used_ports": "0",
                        "used_models": "0",
                    }]
                    await r.set('db_gpu', json.dumps(update_data))
            await asyncio.sleep(0.2)
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] Error: {e}')
            await asyncio.sleep(0.2)

async def redis_add(gpu,running_model,port_vllm,port_model,used_ports,used_models):
    try:        
        current_gpu_info = get_gpu_info()
        res_db_gpu = await r.get('db_gpu')
        if res_db_gpu is not None:
            db_gpu = json.loads(res_db_gpu)
            add_data = {
                        "gpu": gpu, 
                        "gpu_info": str(current_gpu_info),
                        "running_model": running_model,
                        "timestamp": str(datetime.now()),
                        "port_vllm": port_vllm,
                        "port_model": port_model,
                        "used_ports": used_ports,
                        "used_models": used_models
                        }
            db_gpu += [add_data]
            await r.set('db_gpu', json.dumps(db_gpu))
            await asyncio.sleep(0.2)
        else:
            update_data = {
                "gpu": 0, 
                "gpu_info": str(current_gpu_info),
                "running_model": "0",
                "timestamp": str(datetime.now()),
                "port_vllm": "0",
                "port_model": "0",
                "used_ports": "0",
                "used_models": "0"
                }
            await r.set('db_gpu', json.dumps(update_data))
            await asyncio.sleep(0.2)   
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        await asyncio.sleep(0.2)  # Wait before retrying

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(redis_timer())
    yield

app = FastAPI(lifespan=lifespan)
client = docker.from_env()
device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])

@app.get("/")
async def root():
    return f'Hello from server {os.getenv("CONTAINER_PORT")}!'



llm_instances = {}
@app.post("/dockerrest")
async def docker_rest(request: Request):
    try:
        req_data = await request.json()
        
        print(f'req_data {req_data}')

        print(f'llm_instances {llm_instances}')
        


        if req_data["req_type"] == "service":
            try:
                print(f'finding vLLM containers to stop to free GPU memory...')
                container_list = client.containers.list(all=True)
                print(f'found total containers: {len(container_list)}')
                # docker_container_list = get_docker_container_list()
                # docker_container_list_running = [c for c in docker_container_list if c["State"]["Status"] == "running"]
                
                # res_container_list = client.containers.list(all=True)
                # return JSONResponse([container.attrs for container in res_container_list])
                
                print(f'mhmmhmhmh')
                vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                print(f'found total vLLM running containers: {len(vllm_containers_running)}')
                while len(vllm_containers_running) > 0:
                    print(f'stopping all vLLM containers...')
                    for vllm_container in vllm_containers_running:
                        print(f'stopping container {vllm_container.name}...')
                        vllm_container.stop()
                        vllm_container.wait()
                    print(f'waiting for containers to stop...')
                    time.sleep(2)
                    vllm_containers_running = [c for c in container_list if c.name.startswith("vllm") and c.status == "running"]
                print(f'all vLLM containers stopped successfully')                
                
                print(f'loading new model ..')
                container_name = str(req_data["req_model"]).replace('/', '_')
                
                
                res_container = client.containers.run(
                    image="vllm/vllm-openai:latest",
                    name="vllm_bla",
                    runtime="nvidia",
                    ports={"1370/tcp": 1370},
                    volumes={
                        "./vllm/logs": {'bind': '/var/log', 'mode': 'rw'},
                        "/models": {'bind': '/models', 'mode': 'rw'}
                    },
                    shm_size="4g",
                    environment={
                        "NCCL_DEBUG": "INFO"
                    },
                    command=[
                        "--model", "facebook/opt-125m",
                        "--port", "1370",
                        "--tensor-parallel-size", "2",
                        "--gpu_memory_utilization", "0.95",
                        "--max-model-len", "4096"
                    ],
                    detach=True
                )
                                
                                
                                
                
                # res_container = client.containers.run(
                #     "vllm/vllm-openai:latest",
                #     command=f'--model {req_data["req_model"]}',
                #     name=container_name,
                #     runtime=req_data["req_runtime"],
                #     volumes={"/models": {"bind": "/models", "mode": "rw"}},
                #     ports={
                #         f'{req_data["req_port_vllm"]}/tcp': ("0.0.0.0", req_data["req_port_model"])
                #     },
                #     ipc_mode="host",
                #     device_requests=[device_request],
                #     detach=True
                # )
                container_id = res_container.id
                return JSONResponse({"result_status": 200, "result_data": str(container_id)})
                

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 500, "result_data": str(e)})


        if req_data["req_type"] == "update":
            try:

                print(f'trying to update {req_data["req_model"]}...')
                model_id_path = str(req_data["req_model"]).replace('/', '_')
                print(f'model_id_path {model_id_path}...')
                
                selected_model_id_arr = str(req_data["req_model"]).split('/')
                print(f'selected_model_id_arr {selected_model_id_arr}...')
                
                model_id_path_default = f'models--{selected_model_id_arr[0]}--{selected_model_id_arr[1]}'
                print(f'model_id_path_default {model_id_path_default}...')
                
                global llm
                # Unload the previous model
                if 'llm' in globals():
                    print('Unloading the previous model...')
                    del llm  # Delete the LLM object
                    torch.cuda.empty_cache()  # Free GPU memory


                                
                # Check available GPU memory
                free_memory = torch.cuda.mem_get_info()[0] / (1024 ** 3)  # Free memory in GB
                print(f'Available GPU memory: {free_memory:.2f} GB')

                # Estimate model size
                model = AutoModelForCausalLM.from_pretrained(req_data["req_model"], torch_dtype="auto")
                model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)  # Size in GB
                print(f'Model size: {model_size:.2f} GB')

                if model_size > free_memory:                    
                    return JSONResponse({"result_status": 404, "result_data": "Model is too large for available GPU memory"})

                        
                # Load the new model
                print(f'Loading the new model: {req_data["req_model"]}...')
                llm = LLM(
                    model=req_data["req_model"],
                    tensor_parallel_size=2,  # Match the tensor-parallel-size in your Docker config
                    gpu_memory_utilization=0.92  # Match the gpu_memory_utilization in your Docker config
                )

                
                return JSONResponse({"result_status": 200, "result_data": "Model updated successfully"})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})
            
        if req_data["req_type"] == "generate":
            try:
                
                
                print(f'trying to generate ..')
                prompts = [
                    "Hello, my name is",
                    "The president of the United States is",
                    "The capital of France is",
                    "The future of AI is",
                ]
                print(f'trying to generate .. prompts: {prompts}')
                sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
                print(f'trying to generate .. prompts: {sampling_params}')
                llm = LLM(model="facebook/opt-125m")
                print(f'trying to generate .. llm: {llm}')
                outputs = llm.generate(prompts, sampling_params)
                print(f'trying to generate .. outputs: {outputs}')
                for output in outputs:
                    prompt = output.prompt
                    generated_text = output.outputs[0].text
                    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
                                
                return JSONResponse({"result_status": 200, "result_data": outputs[0].outputs[0].text})
                                
                                
                                
                
                
                
                
                
                
                # if req_data["req_model"] not in llm_instances:
                #     print(f'MODEL NOT IN STANCES EGAAAAAAAL')
                # llm_instances[req_data["req_model"]] = LLM(model=req_data["req_model"], task=req_data["req_task"])
                
                # print("llm_instances")
                # print(llm_instances)
                # sampling_params = SamplingParams(int(temperature=req_data["req_temperature"]), top_p=0.9, max_tokens=100)
                
                # outputs = llm_instances[req_data["req_model"]].generate([req_data["req_prompt"]], sampling_params)
                # return JSONResponse({"result_status": 200, "result_data": outputs[0].outputs[0].text})
                
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')                
                return JSONResponse({"result_status": 500, "result_data": str(e)})

        if req_data["req_type"] == "load":
            try:
                llm_instances[req_data["req_model"]] = LLM(model=req_data["req_model"], task=req_data["req_task"])
                
                print("llm_instances")
                print(llm_instances)
                
                return {"message": f'Model {req_data["req_model"]} loaded successfully for instance {req_data["req_model"]}'}
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')  
                return JSONResponse({"result_status": 500, "result_data": e})

        
                
        if req_data["req_type"] == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            return JSONResponse({"result_status": 200, "result_data": res_logs_str})

        if req_data["req_type"] == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result_status": 200, "result_data": stats})

        if req_data["req_type"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["req_type"] == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result_status": 200})

        if req_data["req_type"] == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result_status": 200})

        if req_data["req_type"] == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result_status": 200})

        if req_data["req_type"] == "create":
            try:
                container_name = str(req_data["req_model"]).replace('/', '_')
                res_db_gpu = await r.get('db_gpu')
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)                    
                                        
                    # check if model already downloaded/downloading
                    all_used_models = [g["used_models"] for g in db_gpu]
                    print(f'all_used_models {all_used_models}')
                    if req_data["req_model"] in all_used_models:
                        return JSONResponse({"result_status": 302, "result_data": "Model already downloaded. Trying to start container ..."})
                    
                    # check if ports already used
                    all_used_ports = [g["used_ports"] for g in db_gpu]
                    print(f'all_used_ports {all_used_ports}')
                    if req_data["req_port_vllm"] in all_used_ports or req_data["req_port_model"] in all_used_ports:
                        return JSONResponse({"result_status": 409, "result_data": "Error: Port already in use"})
                    
                    # check if memory available
                    current_gpu_info = get_gpu_info()
                    if current_gpu_info[0]["mem_util"] > 50:
                        all_running_models = [g["running_model"] for g in db_gpu]
                        print(f'all_running_models {all_running_models}')
                        for running_model in all_running_models:
                            req_container = client.containers.get(req_data["req_model"])
                            req_container.stop()
                        
                    # wait for containers to stop
                    for i in range(10):
                        current_gpu_info = get_gpu_info()
                        if current_gpu_info[0]["mem_util"] <= 80:
                            continue
                        else:
                            if i == 9:
                                return JSONResponse({"result_status": 500, "result_data": "Error: Memory > 80%"})
                            else:
                                time.sleep(1)
                    
                    # get all used ports
                    all_used_ports += [req_data["req_port_vllm"],req_data["req_port_model"]]
                    all_used_models += [req_data["req_port_model"]]
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now()),
                        "port_vllm": req_data["req_port_vllm"],
                        "port_model": req_data["req_port_model"],
                        "used_ports": str(all_used_ports),
                        "used_models": str(all_used_models)
                    }
                    
                    db_gpu += [add_data]
                    await r.set('db_gpu', json.dumps(db_gpu))                
                
                else:
                    add_data = {
                        "gpu": 0, 
                        "gpu_info": "0",
                        "running_model": str(container_name),
                        "timestamp": str(datetime.now()),
                        "port_vllm": str(req_data["req_port_vllm"]),
                        "port_model": str(req_data["req_port_model"]),
                        "used_ports": f'{str(req_data["req_port_vllm"])},{str(req_data["req_port_model"])}',
                        "used_models": str(str(req_data["req_model"]))
                    }
                    await r.set('db_gpu', json.dumps(add_data))
                        
                                
                res_container = client.containers.run(
                    "vllm/vllm-openai:latest",
                    command=f'--model {req_data["req_model"]}',
                    name=container_name,
                    runtime=req_data["req_runtime"],
                    volumes={"/home/cloud/.cache/huggingface": {"bind": "/root/.cache/huggingface", "mode": "rw"}},
                    ports={
                        f'{req_data["req_port_vllm"]}/tcp': ("0.0.0.0", req_data["req_port_model"])
                    },
                    ipc_mode="host",
                    device_requests=[device_request],
                    detach=True
                )
                container_id = res_container.id
                return JSONResponse({"result_status": 200, "result_data": str(container_id)})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                r.delete(f'running_model:{str(req_data["req_model"])}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})




        if req_data["req_type"] == "change":
            try:
                print("change stopping .-..")
                req_container = client.containers.get("container_vllm")
                print("got vllm???")
                req_container.stop()
                req_container.wait()
                print("change wait done ..")
                inspect_data = req_container.attrs
                command = inspect_data['Config']['Cmd']
                command_str = " ".join(command)

                command_list = command_str.split()

                try:
                    model_index = command_list.index("--model") + 1
                    command_list[model_index] = "1370"
                except ValueError:
                    print("Error: --model not found in command")
                    return

                try:
                    tp_index = command_list.index("--tensor-parallel-size") + 1
                    command_list[tp_index] = str(2)
                except ValueError:
                    print("Error: --tensor-parallel-size not found in command")
                    return

                new_command = " ".join(command_list)

                req_container.start(command=new_command)
                print(f"Container '{container_name}' restarted with new settings.")

                return JSONResponse({"result_status": 200, "result_data": str(f"Container '{container_name}' restarted with new settings.")})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                r.delete(f'running_model:{str(req_data["req_model"])}')
                return JSONResponse({"result_status": 404, "result_data": str(e)})




    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result_status": 500, "result_data": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))