import os
import requests
import json
import subprocess
import sys
import ast
import time
from datetime import datetime
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import huggingface_hub
import redis
import gradio as gr

current_models_data = []
db_gpu_data = []
db_gpu_data_len = ''

try:
    r = redis.Redis(host="redis", port=6379, db=0)
    db_gpu = json.loads(r.get('db_gpu'))
    print(f'db_gpu: {db_gpu} {len(db_gpu)}')
    db_gpu_data_len = len(db_gpu_data)
except Exception as e:
    print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')


def get_gpu_data():
    try:
        res_gpu_data_all = json.loads(r.get('db_gpu'))
        res_gpu_data_current = res_gpu_data_all[0]
        return res_gpu_data_all
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return e
    
def get_docker_container_list():
    response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"list"})
    res_json = response.json()

    if response.status_code == 200:
        return res_json
    else:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'Error: {response.status_code}'

docker_container_list = get_docker_container_list()

def docker_api_logs(req_model):
    try:
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"logs","req_model":req_model})
        res_json = response.json()
        return f'{res_json["result_data"]}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop'

def docker_api_network(req_container_name):
    try:
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"network","req_container_name":req_container_name})
        res_json = response.json()
        if res_json["result"] == 200:
            return f'{res_json["result_data"]["networks"]["eth0"]["rx_bytes"]}'
        else:
            return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {res_json}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action network {e}'
    
def docker_api_start(req_model):
    try:
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"start","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action start {e}'

def docker_api_stop(req_model):
    try:
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"stop","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action stop {e}'

def docker_api_delete(req_model):
    try:
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"delete","req_model":req_model})
        res_json = response.json()
        return res_json
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] error action delete {e}'

def docker_api_create(req_model, req_pipeline_tag, req_port_model, req_port_vllm):
    try:
        req_container_name = str(req_model).replace('/', '_')
        response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method":"create","req_container_name":req_container_name,"req_model":req_model,"req_runtime":"nvidia","req_port_model":req_port_model,"req_port_vllm":req_port_vllm})
        response_json = response.json()
        
        new_entry = [{
            "gpu": 8,
            "path": f'/home/cloud/.cache/huggingface/{req_model}',
            "container": "0",
            "container_status": "0",
            "running_model": req_container_name,
            "model": req_model,
            "pipeline_tag": req_pipeline_tag,
            "port_model": req_port_model,
            "port_vllm": req_port_vllm
        }]
        r.set("db_gpu", json.dumps(new_entry))

        print(response_json["result"])
        if response_json["result"] == 200:
            return f'{response_json["result_data"]}'
        else:
            return f'Create result ERR no container_id: {str(response_json)}'
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'error docker_api_create'

def search_models(query):
    try:
        global current_models_data    
        response = requests.get(f'https://huggingface.co/api/models?search={query}')
        response_models = response.json()
        current_models_data = response_models.copy()
        model_ids = [m["id"] for m in response_models]
        return gr.update(choices=model_ids, value=response_models[0]["id"], show_label=True, label=f'found {len(response_models)} models!')
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')

def calculate_model_size(json_info): # to fix    
    try:
        d_model = json_info.get("hidden_size") or json_info.get("d_model")
        num_hidden_layers = json_info.get("num_hidden_layers", 0)
        num_attention_heads = json_info.get("num_attention_heads") or json_info.get("decoder_attention_heads") or json_info.get("encoder_attention_heads", 0)
        intermediate_size = json_info.get("intermediate_size") or json_info.get("encoder_ffn_dim") or json_info.get("decoder_ffn_dim", 0)
        vocab_size = json_info.get("vocab_size", 0)
        num_channels = json_info.get("num_channels", 3)
        patch_size = json_info.get("patch_size", 16)
        torch_dtype = json_info.get("torch_dtype", "float32")
        bytes_per_param = 2 if torch_dtype == "float16" else 4
        total_size_in_bytes = 0
        
        if json_info.get("model_type") == "vit":
            embedding_size = num_channels * patch_size * patch_size * d_model
            total_size_in_bytes += embedding_size

        if vocab_size and d_model:
            embedding_size = vocab_size * d_model
            total_size_in_bytes += embedding_size

        if num_attention_heads and d_model and intermediate_size:
            attention_weights_size = num_hidden_layers * (d_model * d_model * 3)
            ffn_weights_size = num_hidden_layers * (d_model * intermediate_size + intermediate_size * d_model)
            layer_norm_weights_size = num_hidden_layers * (2 * d_model)

            total_size_in_bytes += (attention_weights_size + ffn_weights_size + layer_norm_weights_size)

        if json_info.get("is_encoder_decoder"):
            encoder_size = num_hidden_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            decoder_layers = json_info.get("decoder_layers", 0)
            decoder_size = decoder_layers * (num_attention_heads * d_model * d_model + intermediate_size * d_model + d_model * intermediate_size + 2 * d_model)
            
            total_size_in_bytes += (encoder_size + decoder_size)

        return total_size_in_bytes
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return 0

def get_info(selected_id):    
    global current_models_data    
    res_model_data = {
        "search_data" : "",
        "model_id" : selected_id,
        "pipeline_tag" : "",
        "transformers" : "",
        "private" : "",
        "downloads" : ""
    }

    try:
        for item in current_models_data:
            if item['id'] == selected_id:
                
                res_model_data["search_data"] = item
                
                if "pipeline_tag" in item:
                    res_model_data["pipeline_tag"] = item["pipeline_tag"]
  
                if "tags" in item:
                    if "transformers" in item["tags"]:
                        res_model_data["transformers"] = True
                    else:
                        res_model_data["transformers"] = False
                                    
                if "private" in item:
                    res_model_data["private"] = item["private"]
                                  
                if "downloads" in item:
                    res_model_data["downloads"] = item["downloads"]
                  
                container_name = str(res_model_data["model_id"]).replace('/', '_')
                
                return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"], container_name
                
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return res_model_data["search_data"], res_model_data["model_id"], res_model_data["pipeline_tag"], res_model_data["transformers"], res_model_data["private"], res_model_data["downloads"]

def get_additional_info(selected_id):    
        res_model_data = {
            "hf_data" : "",
            "config_data" : "",
            "model_id" : selected_id,
            "size" : 0,
            "gated" : ""
        }                
        try:
            try:
                model_info = huggingface_hub.model_info(selected_id)
                model_info_json = vars(model_info)
                res_model_data["hf_data"] = model_info_json
                
                if "gated" in model_info.__dict__:
                    res_model_data['gated'] = model_info_json["gated"]
                
                if "safetensors" in model_info.__dict__:                        
                    safetensors_json = vars(model_info.safetensors)
                    res_model_data['size'] = safetensors_json["total"]
            
            except Exception as get_model_info_err:
                res_model_data['hf_data'] = f'{get_model_info_err}'
                pass
                    
            try:
                url = f'https://huggingface.co/{selected_id}/resolve/main/config.json'
                response = requests.get(url)
                if response.status_code == 200:
                    response_json = response.json()
                    res_model_data["config_data"] = response_json                     
                else:
                    res_model_data["config_data"] = f'{response.status_code}'
                    
            except Exception as get_config_json_err:
                res_model_data["config_data"] = f'{get_config_json_err}'
                pass                       
            
            if res_model_data["size"] == 0:
                try:
                    res_model_data["size"] = calculate_model_size(res_model_data["config_data"]) 
                except Exception as get_config_json_err:
                    res_model_data["size"] = 0
    
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]
    
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return res_model_data["hf_data"], res_model_data["config_data"], res_model_data["model_id"], res_model_data["size"], res_model_data["gated"]

def gr_load_check(selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated):
    if selected_model_pipeline_tag != '' and selected_model_transformers == 'True' and selected_model_private == 'False' and selected_model_gated == 'False':
        return gr.update(visible=False), gr.update(value=f'Download {selected_model_id[:12]}...', visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

rx_change_arr = []
def check_rx_change(current_rx_bytes):
    try:                
        try:
            int(current_rx_bytes)
        except ValueError:
            return '0'
        global rx_change_arr
        rx_change_arr += [int(current_rx_bytes)]
        if len(rx_change_arr) > 4:
            last_value = rx_change_arr[-1]
            same_value_count = 0
            for i in range(1,len(rx_change_arr)):
                if rx_change_arr[i*-1] == last_value:
                    same_value_count += 1
                    if same_value_count > 10:
                        return f'Count > 10 Download finished'
                else:
                    return f'Count: {same_value_count} {str(rx_change_arr)}'
            return f'Count: {same_value_count} {str(rx_change_arr)}'        
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return f'Error rx_change_arr {str(e)}'

def json_to_pd():       
    rows = []
    try:
        gpu_list = get_gpu_data()
                
        for entry in gpu_list:
            gpu_info = ast.literal_eval(entry['gpu_info'])[0]
            rows.append({
                "#": entry["gpu"],
                "GPU Usage": f'{gpu_info["gpu_util"]} %',
                "Memory Usage": f'{"{:.2f}".format(gpu_info["mem_util"])} % ({gpu_info["mem_used"]} MB/{gpu_info["mem_total"]} MB)',
                "Running model": entry["running_model"],
                "Timestamp": entry["timestamp"],
                "Port vLLM": entry["port_vllm"],
                "Port model": entry["port_model"],
                "Used ports": entry["used_ports"],
                "Used models": entry["used_models"]                
            })
        df = pd.DataFrame(rows)
        return df
    
    except Exception as e:
        print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
        return rows

with gr.Blocks() as app:
    gr.Markdown(
        """
        # Welcome!
        Select a Hugging Face model and deploy it on a port
        
        **Note**: _[vLLM supported models list](https://docs.vllm.ai/en/latest/models/supported_models.html)_        
        """)

    inp = gr.Textbox(placeholder="Type in a Hugging Face model or tag", show_label=False, autofocus=True)
    btn = gr.Button("Search")
    out = gr.Textbox(visible=False)

    model_dropdown = gr.Dropdown(choices=[''], interactive=True, show_label=False, visible=False)

    with gr.Row():
        selected_model_id = gr.Textbox(label="id",visible=False)
        selected_model_container_name = gr.Textbox(label="container_name",visible=False)
        
    with gr.Row():       
        selected_model_pipeline_tag = gr.Textbox(label="pipeline_tag", visible=False)
        selected_model_transformers = gr.Textbox(label="transformers", visible=False)
        selected_model_private = gr.Textbox(label="private", visible=False)
        
    with gr.Row():
        selected_model_size = gr.Textbox(label="size", visible=False)
        selected_model_gated = gr.Textbox(label="gated", visible=False)
        selected_model_downloads = gr.Textbox(label="downloads", visible=False)
    
    selected_model_search_data = gr.Textbox(label="search_data", visible=False)
    selected_model_hf_data = gr.Textbox(label="hf_data", visible=False)
    selected_model_config_data = gr.Textbox(label="config_data", visible=False)
    gr.Markdown(
        """
        <hr>
        """
    )            

    inp.submit(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)
    btn.click(search_models, inputs=inp, outputs=[model_dropdown]).then(lambda: gr.update(visible=True), None, model_dropdown)

    with gr.Row():
        port_model = gr.Number(value=8001,visible=False,label="Port of model: ")
        port_vllm = gr.Number(value=8000,visible=False,label="Port of vLLM: ")
    
    info_textbox = gr.Textbox(value="Interface not possible for selected model. Try another model or check 'pipeline_tag', 'transformers', 'private', 'gated'", show_label=False, visible=False)
    btn_dl = gr.Button("Download", visible=False)
    
    model_dropdown.change(get_info, model_dropdown, [selected_model_search_data,selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_downloads],selected_model_container_name).then(get_additional_info, model_dropdown, [selected_model_hf_data, selected_model_config_data, selected_model_id, selected_model_size, selected_model_gated]).then(lambda: gr.update(visible=True), None, selected_model_pipeline_tag).then(lambda: gr.update(visible=True), None, selected_model_transformers).then(lambda: gr.update(visible=True), None, selected_model_private).then(lambda: gr.update(visible=True), None, selected_model_downloads).then(lambda: gr.update(visible=True), None, selected_model_size).then(lambda: gr.update(visible=True), None, selected_model_gated).then(lambda: gr.update(visible=True), None, port_model).then(lambda current_value: current_value + 1, port_model, port_model).then(lambda: gr.update(visible=True), None, port_vllm).then(lambda current_value: current_value + 1, port_vllm, port_vllm).then(gr_load_check, [selected_model_id,selected_model_pipeline_tag,selected_model_transformers,selected_model_private,selected_model_gated],[info_textbox,btn_dl])

    create_response = gr.Textbox(label="Building container...", show_label=True, visible=False)  
    timer_dl_box = gr.Textbox(label="Dowmload progress:", visible=False)
    
    btn_interface = gr.Button("Load Interface",visible=False)
    @gr.render(inputs=[selected_model_pipeline_tag, selected_model_id], triggers=[btn_interface.click])
    def show_split(text_pipeline, text_model):
        if len(text_model) == 0:
            gr.Markdown("Error pipeline_tag or model_id")
        else:
            gr.Interface.from_pipeline(pipeline(text_pipeline, model=text_model))

    gpu_dataframe = gr.Dataframe(label="GPU information")
    gpu_timer = gr.Timer(1,active=True)
    gpu_timer.tick(json_to_pd, outputs=gpu_dataframe)
    container_state = gr.State([])   
    docker_container_list = get_docker_container_list()     
    @gr.render(inputs=container_state)
    def render_container(render_container_list):
        docker_container_list = get_docker_container_list()
        docker_container_list_running = [c for c in docker_container_list if c["State"]["Status"] == "running"]
        docker_container_list_not_running = [c for c in docker_container_list if c["State"]["Status"] != "running"]

        def refresh_container():
            try:
                global docker_container_list
                response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
                docker_container_list = response.json()
                return docker_container_list
            
            except Exception as e:
                print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
                return f'err {str(e)}'
            
        gr.Markdown(f'### Container running ({len(docker_container_list_running)})')

        for current_container in docker_container_list_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container Id")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
                
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)

            with gr.Row():            
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)     
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )

                stop_btn = gr.Button("Stop", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                stop_btn.click(
                    docker_api_stop,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
                
            gr.Markdown(
                """
                <hr>
                """
            )


        gr.Markdown(f'### Container not running ({len(docker_container_list_not_running)})')

        for current_container in docker_container_list_not_running:
            with gr.Row():
                
                container_id = gr.Textbox(value=current_container["Id"][:12], interactive=False, elem_classes="table-cell", label="Container ID")
                
                container_name = gr.Textbox(value=current_container["Name"][1:], interactive=False, elem_classes="table-cell", label="Container Name")              
    
                container_status = gr.Textbox(value=current_container["State"]["Status"], interactive=False, elem_classes="table-cell", label="Status")
                
                container_ports = gr.Textbox(value=next(iter(current_container["HostConfig"]["PortBindings"])), interactive=False, elem_classes="table-cell", label="Port")
            
            with gr.Row():
                container_log_out = gr.Textbox(value=[], lines=20, interactive=False, elem_classes="table-cell", show_label=False, visible=False)
                
            with gr.Row():
                logs_btn = gr.Button("Show Logs", scale=0)
                logs_btn_close = gr.Button("Close Logs", scale=0, visible=False)
                
                logs_btn.click(
                    docker_api_logs,
                    inputs=[container_id],
                    outputs=[container_log_out]
                ).then(
                    lambda :[gr.update(visible=False), gr.update(visible=True), gr.update(visible=True)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                
                logs_btn_close.click(
                    lambda :[gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)], None, [logs_btn,logs_btn_close, container_log_out]
                )
                                
                start_btn = gr.Button("Start", scale=0)
                delete_btn = gr.Button("Delete", scale=0, variant="stop")

                start_btn.click(
                    docker_api_start,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )

                delete_btn.click(
                    docker_api_delete,
                    inputs=[container_id],
                    outputs=[container_state]
                ).then(
                    refresh_container,
                    outputs=[container_state]
                )
            
            gr.Markdown(
                """
                <hr>
                """
            )
            
    def refresh_container_list():
        try:
            global docker_container_list
            response = requests.post(f'http://container_backend:{str(int(os.getenv("CONTAINER_PORT"))+1)}/dockerrest', json={"req_method": "list"})
            docker_container_list = response.json()
            return docker_container_list
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return f'err {str(e)}'
                 
    def check_container_running(container_name):
        try:
            docker_container_list = get_docker_container_list()
            docker_container_list_running = [c for c in docker_container_list if c["name"] == container_name]
            if len(docker_container_list_running) > 0:
                return f'Yes container is running!'
        except Exception as e:
            print(f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {e}')
            return f'err {str(e)}'
    
    timer_dl = gr.Timer(1,active=False)
    timer_dl.tick(docker_api_network, create_response, timer_dl_box)
    
    btn_dl.click(lambda: gr.update(label="Building vLLM container",visible=True), None, create_response).then(docker_api_create,inputs=[model_dropdown,selected_model_pipeline_tag,port_model,port_vllm],outputs=create_response).then(refresh_container_list, outputs=[container_state]).then(lambda: gr.Timer(active=True), None, timer_dl).then(lambda: gr.update(visible=True), None, timer_dl_box).then(lambda: gr.update(visible=True), None, btn_interface)

app.launch(server_name="0.0.0.0", server_port=int(os.getenv("CONTAINER_PORT")))