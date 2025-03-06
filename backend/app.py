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
import psutil

# Initialize Redis connection
r = redis.Redis(host="redis", port=6379, db=0)

# Initialize NVML for GPU monitoring
pynvml.nvmlInit()

# Initialize Docker client
client = docker.from_env()

# Global variables for network monitoring
prev_bytes_recv = 0
rx_change_arr = []


def get_gpu_info():
    """Retrieve GPU utilization and memory information."""
    try:
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util = utilization.gpu
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = memory_info.used / 1024**2
            mem_total = memory_info.total / 1024**2
            mem_util = (mem_used / mem_total) * 100

            gpu_info.append({
                "gpu_id": i,
                "gpu_util": float(gpu_util),
                "mem_used": float(mem_used),
                "mem_total": float(mem_total),
                "mem_util": float(mem_util),
                "timestamp": str(datetime.now())
            })
        return gpu_info
    except Exception as e:
        print(f"Error retrieving GPU info: {e}")
        return []


def get_download_speed():
    """Retrieve network download speed and total bytes received."""
    try:
        global prev_bytes_recv
        global rx_change_arr

        net_io = psutil.net_io_counters()
        bytes_recv = net_io.bytes_recv
        download_speed = bytes_recv - prev_bytes_recv
        prev_bytes_recv = bytes_recv
        download_speed_mbit_s = (download_speed * 8) / (1024 ** 2)
        bytes_received_mb = bytes_recv / (1024 ** 2)
        rx_change_arr.append(download_speed_mbit_s)
        return {
            "download_speed_mbit_s": f"{download_speed_mbit_s:.2f}",
            "bytes_received_mb": f"{bytes_received_mb:.2f}",
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        print(f"Error retrieving network info: {e}")
        return {"error": str(e)}


def get_container_network_stats(container):
    """Retrieve network statistics for a Docker container."""
    try:
        stats = container.stats(stream=False)
        network_stats = stats.get("networks", {})
        return {
            "container_id": container.id,
            "network_stats": network_stats,
            "timestamp": str(datetime.now())
        }
    except Exception as e:
        print(f"Error retrieving container network stats: {e}")
        return {"error": str(e)}


async def redis_timer():
    """Periodically update Redis with GPU, Docker, and network information."""
    while True:
        try:
            # Update GPU info
            current_gpu_info = get_gpu_info()
            await r.set("gpu_info", json.dumps(current_gpu_info))

            # Update Docker container info
            container_data = []
            containers = client.containers.list(all=True)
            for container in containers:
                container_info = {
                    "container_id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "network_stats": get_container_network_stats(container),
                    "timestamp": str(datetime.now())
                }
                container_data.append(container_info)
            await r.set("docker_container_info", json.dumps(container_data))

            # Update network info
            network_data = get_download_speed()
            await r.set("network_info", json.dumps(network_data))

            await asyncio.sleep(0.2)
        except Exception as e:
            print(f"Error in redis_timer: {e}")
            await asyncio.sleep(0.2)


async def redis_add(gpu, running_model, port_vllm, port_model, used_ports, used_models):
    """Add GPU and container information to Redis."""
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
    except Exception as e:
        print(f"Error in redis_add: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start background tasks when the app starts."""
    asyncio.create_task(redis_timer())
    yield


app = FastAPI(lifespan=lifespan)
device_request = DeviceRequest(count=-1, capabilities=[["gpu"]])


@app.get("/")
async def root():
    return f'Hello from server {os.getenv("CONTAINER_PORT")}!'


@app.post("/dockerrest")
async def docker_rest(request: Request):
    try:
        req_data = await request.json()

        if req_data["req_method"] == "test":
            print(f'got test!')
            print("req_data")
            print(req_data)
            return JSONResponse({"result": 200, "result_data": "teeest okoko"})

        if req_data["req_method"] == "logs":
            req_container = client.containers.get(req_data["req_model"])
            res_logs = req_container.logs()
            res_logs_str = res_logs.decode('utf-8')
            return JSONResponse({"result": 200, "result_data": res_logs_str})

        if req_data["req_method"] == "network":
            req_container = client.containers.get(req_data["req_container_name"])
            stats = req_container.stats(stream=False)
            return JSONResponse({"result": 200, "result_data": stats})

        if req_data["req_method"] == "list":
            res_container_list = client.containers.list(all=True)
            return JSONResponse([container.attrs for container in res_container_list])

        if req_data["req_method"] == "delete":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            req_container.remove(force=True)
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "stop":
            req_container = client.containers.get(req_data["req_model"])
            req_container.stop()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "start":
            req_container = client.containers.get(req_data["req_model"])
            req_container.start()
            return JSONResponse({"result": 200})

        if req_data["req_method"] == "create":
            try:
                container_name = str(req_data["req_model"]).replace('/', '_')
                container_name = f'vllm_{container_name}'
                res_db_gpu = await r.get('db_gpu')
                if res_db_gpu is not None:
                    db_gpu = json.loads(res_db_gpu)

                    # Check if model already downloaded/downloading
                    all_used_models = [g["used_models"] for g in db_gpu]
                    print(f'all_used_models {all_used_models}')
                    if req_data["req_model"] in all_used_models:
                        return JSONResponse({"result": 302, "result_data": "Model already downloaded. Trying to start container ..."})

                    # Check if ports already used
                    all_used_ports = [g["used_ports"] for g in db_gpu]
                    print(f'all_used_ports {all_used_ports}')
                    if req_data["req_port_vllm"] in all_used_ports or req_data["req_port_model"] in all_used_ports:
                        return JSONResponse({"result": 409, "result_data": "Error: Port already in use"})

                    # Check if memory available
                    current_gpu_info = get_gpu_info()
                    if current_gpu_info[0]["mem_util"] > 50:
                        all_running_models = [g["running_model"] for g in db_gpu]
                        print(f'all_running_models {all_running_models}')
                        for running_model in all_running_models:
                            req_container = client.containers.get(req_data["req_model"])
                            req_container.stop()

                    # Wait for containers to stop
                    for i in range(10):
                        current_gpu_info = get_gpu_info()
                        if current_gpu_info[0]["mem_util"] <= 80:
                            continue
                        else:
                            if i == 9:
                                return JSONResponse({"result": 500, "result_data": "Error: Memory > 80%"})
                            else:
                                time.sleep(1)

                    # Get all used ports
                    all_used_ports += [req_data["req_port_vllm"], req_data["req_port_model"]]
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

                print(f'finding containers to stop to free GPU memory...')
                container_list = client.containers.list(all=True)
                print(f'found total containers: {len(container_list)}')

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

                res_container = client.containers.run(
                    "vllm/vllm-openai:latest",
                    command=f'--model {req_data["req_model"]} --tensor-parallel-size 2',
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
                return JSONResponse({"result": 200, "result_data": str(container_id)})

            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                await r.delete(f'running_model:{str(req_data["req_model"])}')
                return JSONResponse({"result": 404, "result_data": str(e)})

    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return JSONResponse({"result": 500, "result_data": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("CONTAINER_PORT")))