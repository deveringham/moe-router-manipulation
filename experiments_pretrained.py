###
# experiments_pretrained.py
#
# Routines for MoE experiments on pretrained models from Huggingface.
# Dylan Everingham
# 30.04.2026
###

import torch
import time
import datetime
import h5py
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from monitoring import *

routing_data_dir = "./routing_logs/"

def load_tokenizer_qwen_bnb():
    return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat",
                                         trust_remote_code=True)

def load_model_qwen_bnb():
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        device_map="auto",
        #dtype=torch.float16,
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    tokenizer = load_tokenizer_qwen_bnb()
    return model, tokenizer

def load_tokenizer_qwen_gptq():
    return AutoTokenizer.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
                                         trust_remote_code=True)

def load_model_qwen_gptq():

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = load_tokenizer_qwen_gptq()
    return model, tokenizer

def load_tokenizer_deepseek_bnb():
    return AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite-Chat",
                                         trust_remote_code=True)

def load_model_deepseek_bnb():
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-V2-Lite-Chat",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    tokenizer = load_tokenizer_deepseek_bnb()
    return model, tokenizer

def load_tokenizer_mistral_bnb():
    return AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1",
                                         trust_remote_code=True)

def load_model_mistral_bnb():
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    tokenizer = load_tokenizer_mistral_bnb()
    return model, tokenizer

def chat_generate(model, tokenizer, prompt="", max_new_tokens=100):
        
    # Set chat template and transfer to device
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt")

    # Generate output
    generated_ids = model.generate(
        model_inputs['input_ids'],
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode and return
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


def single_generate(model, tokenizer, moe_probe, prompt="", max_new_tokens=100):

    # Clear monitoring probe
    moe_probe.clear()
        
    response = chat_generate(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
    
    # Get metrics
    probs = moe_probe.get_probs()
    active_experts = moe_probe.get_active_experts()
    
    return response, probs, active_experts


def single_generate_noprobe(model, tokenizer, prompt="", max_new_tokens=100):
        
    response = chat_generate(model, tokenizer, prompt=prompt, max_new_tokens=max_new_tokens)
    
    return response

def save_eam_data(filename, run_id, eam_results):
    with h5py.File(filename, 'w') as f:
        
        # For each sample...
        count = 0
        for sample in eam_results:
            
            # Store EAM
            eam_dataset = f.create_dataset(
                f"eam_{count}", 
                data=sample['eam'].cpu().numpy(), 
                compression="gzip"
            )
            
            # Sore run id
            f.attrs['run_id'] = run_id

            # Store all other metrics
            for key, value in sample['metrics'].items():
                eam_dataset.attrs[key] = value
            
            count += 1

def save_routing_data(filename, run_id, results):
    
    with h5py.File(filename, 'w') as f:
        
        # For each sample...
        count = 0
        for sample in results:
            
            # Store metrics
            dataset_probs = f.create_dataset(
                f"probs_{count}",
                data=sample['probs'].cpu().numpy(), 
                compression="gzip"
            )
            dataset_active_experts = f.create_dataset(
                f"active_experts_{count}",
                data=sample['active_experts'].cpu().numpy(), 
                compression="gzip"
            )
            
            # Store run id
            f.attrs['run_id'] = run_id

            # Store all other metrics as attributes of prob dataset
            for key, value in sample['metrics'].items():
                dataset_probs.attrs[key] = value
            
            count += 1
            
def run_experiment_mmlu(model_choice, n_samples, start_sample=0, save_samples=100, 
                        max_new_tokens=100, shuffle_seed=100,
                        save_results=True, no_probe=False):
    
    if no_probe:
        return run_experiment_mmlu_eam_noprobe(model_choice=model_choice, n_samples=n_samples,
                                               start_sample=start_sample, save_samples=save_samples,
                                               max_new_tokens=max_new_tokens, shuffle_seed=shuffle_seed,
                                               save_results=save_results)
    
    # Get model (with attached MoE monitoring probe) and tokenizer
    if model_choice == "qwen":
        model, tokenizer = load_model_qwen()
        probe = MoEProbeQwen(model)
    elif model_choice == "qwen_gptq":
        model, tokenizer = load_model_qwen_gptq()
        probe = MoEProbeQwen(model)
    elif model_choice == "deepseek":
        model, tokenizer = load_model_deepseek()
        probe = MoEProbeDeepSeek(model)
    elif model_choice == "mistral":
        model, tokenizer = load_model_mistral()
        probe = MoEProbeMistral(model)
    else:
        raise ValueError("Invalid model_choice. Select 'qwen_bitsandbytes', 'qwen_gptq', 'deepseek_bitsandbytes_bitsandbytes', or 'mistral_bitsandbytes'.")
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"{model_choice}-{timestamp}-samples{n_samples}-tokens{max_new_tokens}"
    
    # Get data
    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=shuffle_seed)
    
    # Results will contain EAM for each sample plus prompt and response
    results = []
    
    # For each sample...
    count = 0
    for sample in dataset:
        
        count += 1
        
        # Skip to starting sample
        if count >= start_sample:
            print(f"Generating response {count}/{n_samples}...")

            prompt = sample['question']
            
            # Start timing
            start_time = time.perf_counter()

            # Generate response and get metrics
            response, probs, active_experts = single_generate(model, tokenizer, probe,
                                                              prompt=prompt, max_new_tokens=max_new_tokens)
            
            # Stop timing
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.3f}s.")

            # Store results
            result = {}
            result['probs'] = probs
            result['active_experts'] = active_experts
            result['metrics'] = {
                'prompt': prompt,
                'response': response,
                'prompt_tokenized': tokenizer.encode(prompt),
                'response_tokenized': tokenizer.encode(response),
                'subject': sample['subject'],
                'inference_time': inference_time,
            }
            results.append(result)

            # Write results to file
            if ((count % save_samples == 0) and save_results):
                filename = routing_data_dir + run_id + '-n' + str((count//save_samples)-1) + '.h5'
                print("Saving outputs to file " + filename)
                save_routing_data(filename, run_id, results)
                results = []
            
    # Write final results
    if ((count % save_samples) and save_results) != 0:
        filename = routing_data_dir + run_id + '-n' + str(count//save_samples) + '.h5'
        print("Saving outputs to file " + filename)
        save_routing_data(filename, run_id, results)
        results = []

    # If not recording results, simply return them
    if not save_results:
        return results
    
def run_experiment_mmlu_noprobe(model_choice, n_samples, start_sample=0, save_samples=100,
                                max_new_tokens=100, shuffle_seed=100,
                                save_results=True):
    
    # Get model (with attached MoE monitoring probe) and tokenizer
    if model_choice == "qwen_bitsandbytes":
        model, tokenizer = load_model_qwen_bitsandbytes()
    elif model_choice == "qwen_gptq":
        model, tokenizer = load_model_qwen_gptq()
    elif model_choice == "deepseek_bitsandbytes":
        model, tokenizer = load_model_deepseek_bitsandbytes()
    elif model_choice == "mistral_bitsandbytes":
        model, tokenizer = load_model_mistral_bitsandbytes()
    else:
        raise ValueError("Invalid model_choice. Select 'qwen_bitsandbytes', 'qwen_gptq', 'deepseek_bitsandbytes_bitsandbytes', or 'mistral_bitsandbytes'.")
    
    # Get unique string id for the run
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')
    run_id = f"{model_choice}-{timestamp}-samples{n_samples}-tokens{max_new_tokens}"
    
    # Get data
    dataset = get_data_mmlu(n_samples=n_samples, shuffle_seed=shuffle_seed)
    
    # Results will contain EAM for each sample plus prompt and response
    results = []
    
    # For each sample...
    count = 0
    for sample in dataset:
        
        count += 1
        
        # Skip to starting sample
        if count >= start_sample:
            print(f"Generating response {count}/{n_samples}...")

            prompt = sample['question']
            
            # Start timing
            start_time = time.perf_counter()

            # Generate response and get metrics
            response = single_generate_noprobe(model, tokenizer,
                                               prompt=prompt, max_new_tokens=max_new_tokens)
            
            # Stop timing
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            print(f"Inference took {inference_time:.3f}s.")

            # Store results
            result = {}
            result['metrics'] = {
                'prompt': prompt,
                'response': response,
                'prompt_tokenized': tokenizer.encode(prompt),
                'response_tokenized': tokenizer.encode(response),
                'subject': sample['subject'],
                'inference_time': inference_time,
            }
            results.append(result)

            # Write results to file
            if ((count % save_samples == 0) and save_results):
                filename = routing_data_dir + run_id + '-n' + str((count//save_samples)-1) + '.h5'
                print("Saving outputs to file " + filename)
                save_routing_data(filename, run_id, results)
                results = []
            
    # Write final results
    if ((count % save_samples) and save_results) != 0:
        filename = routing_data_dir + run_id + '-n' + str(count//save_samples) + '.h5'
        print("Saving outputs to file " + filename)
        save_routing_data(filename, run_id, results)
        results = []
    
    # If not recording results, simply return them
    if not save_results:
        return results