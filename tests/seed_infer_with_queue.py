# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
import platform
import json
import queue
from tqdm import tqdm
from typing import List, Dict

import torch.multiprocessing as mp
from torch.multiprocessing import Manager
from datasets import load_dataset

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli.SparkTTS import SparkTTS

from py3_tools.py_debug import breakpoint

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="/nfs/pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="example/seedtts-results",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device number")
    parser.add_argument(
        "--text", default='hell, how are you', type=str, required=False, help="Text for TTS generation"
    )
    parser.add_argument("--prompt_text", type=str, help="Transcript of prompt audio")
    parser.add_argument(
        "--prompt_speech_path",
        type=str,
        help="Path to the prompt audio file",
    )
    parser.add_argument("--gender", choices=["male", "female"])
    parser.add_argument(
        "--pitch", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    parser.add_argument(
        "--speed", choices=["very_low", "low", "moderate", "high", "very_high"]
    )
    
    return parser.parse_args()


def process_rank(rank, args, input_queue: mp.Queue, output_queue: mp.Queue):
    """Perform TTS inference using queue communication and save the generated audio."""
    logging.info(f"Rank {rank} using model from: {args.model_dir}")
    logging.info(f"Rank {rank} saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)
    out_wav_dir = os.path.join(args.save_dir, 'wav')
    os.makedirs(out_wav_dir, exist_ok=True)

    # Convert device argument to torch.device
    device = torch.device(f'cuda:{rank}')
    
    # Initialize the model
    model = SparkTTS(args.model_dir, device)

    logging.info(f"Rank {rank} starting inference...")
    
    # Get number of GPUs
    args.ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.ngpu < 1:
        raise ValueError("Number of GPUs (ngpu) must be at least 1.")

    item_count = 0
    
    try:
        # Process items from input queue until we receive None (termination signal)
        while True:
            item = input_queue.get()
            
            # Check for termination signal
            if item is None:
                logging.info(f"Rank {rank} received termination signal")
                break
                
            item_count += 1
            text = item['text_input']
            prompt_text = item['prompt_text']
            prompt_speech_path = item['prompt_audio']
            output_name = item['audio_output_path']

            item['audio_output_path'] = os.path.join(out_wav_dir, output_name)
            save_path = os.path.join(out_wav_dir, output_name)
            save_path = os.path.abspath(save_path)
            
            res = {
                'index': item.get('index', f"rank{rank}_{item_count}"),
                'text': text,
                'wav_path': save_path,
                'ref_wav_path': prompt_speech_path,
                'prompt_text': prompt_text
            }
            
            breakpoint()
            if os.path.exists(save_path):
                logging.info(f"Rank {rank} - File already exists: {save_path}, skipping inference.")
            else:
                # Perform inference and save the output audio
                with torch.no_grad():
                    
                    try:
                        wav = model.inference(
                            text,
                            prompt_speech_path,
                            prompt_text=prompt_text,
                            gender=args.gender,
                            pitch=args.pitch,
                            speed=args.speed,
                        )
                    except Exception as e:
                        logging.error(f"Rank {rank} - Error during inference for text '{text}': {str(e)}", exc_info=True)
                        wav = None
                    
                    if wav is None:
                        res['wav_path'] = ''
                        logging.warning(f"Rank {rank} - Skipping empty output for text: {text}")
                    else:
                        sf.write(save_path, wav, samplerate=16000)
                        logging.info(f"Rank {rank} - Audio saved at: {save_path}")
            
            # Put result in output queue
            output_queue.put(res)
            
    except Exception as e:
        logging.error(f"Rank {rank} encountered error: {str(e)}", exc_info=True)
    finally:
        logging.info(f"Rank {rank} processed {item_count} samples.")


def main(config):
    # 确保多进程启动方式正确
    mp.set_start_method('spawn', force=True)
    
    ## load dataset
    input_dataset_path = '/hy-netdisk/datasets_processed/seedtts_testset'
    dataset = load_dataset(input_dataset_path,**{"name": "en", "split": "test_wer"})
    
    # Convert dataset to list of dicts with proper indexing
    df = dataset.to_pandas()
    df = df.reset_index()  # Add index as a column
    dataset = df.to_dict(orient='records')
    
    ## process dataset
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        raise ValueError("Number of GPUs (ngpu) must be at least 1.")
    
    logging.info(f"Using {world_size} GPUs for processing")
    
    if world_size == 1:
        # If only one GPU, run directly in the main process with simple queues
        print("Running on a single GPU, using simple queue")
        
        # Create simple queues
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Fill input queue
        for item in dataset:
            input_queue.put(item)
        # Add termination signal
        input_queue.put(None)
        
        # Process data
        process_rank(0, config, input_queue, output_queue)
        
        # Collect results
        text_wav_pairs = []
        while not output_queue.empty():
            res = output_queue.get()
            if res is None:
                break
            text_wav_pairs.append(res)
    else:
        # For multiple GPUs, use multiprocessing queues
        with Manager() as manager:
            # Create queues
            input_queue = manager.Queue()
            output_queue = manager.Queue()
            
            # Fill input queue with data
            for item in dataset:
                input_queue.put(item)
            
            # Add termination signals for each process
            for _ in range(world_size):
                input_queue.put(None)
            
            # Start worker processes
            processes = []
            for rank in range(world_size):
                p = mp.Process(
                    target=process_rank,
                    args=(rank, config, input_queue, output_queue)
                )
                p.start()
                processes.append(p)
            
            # Collect results with progress bar
            text_wav_pairs = []
            total_items = len(dataset)
            with tqdm(total=total_items, desc="Collecting results") as pbar:
                while len(text_wav_pairs) < total_items:
                    res = output_queue.get()
                    if res is not None:
                        text_wav_pairs.append(res)
                        pbar.update(1)
            
            # Wait for all processes to finish
            for p in processes:
                p.join()
    
    # Sort results by index to maintain original order
    text_wav_pairs.sort(key=lambda x: x['index'])
    
    # Save results
    output_dir = config.save_dir
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, f'text_wav_{os.path.basename(output_dir)}.json'), 'w') as f:
        json.dump(text_wav_pairs, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Successfully processed {len(text_wav_pairs)} items")
    logging.info(f"Results saved to {output_dir}")
    
    print("Text and wav pairs saved to (len={}, error_wav={}): {} \n".format(
            len(text_wav_pairs),
            sum(1 for item in text_wav_pairs if not item['wav_path']),
            os.path.join(output_dir, f'text_wav_{os.path.basename(output_dir)}.json')
        ))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    
    # switch here to test generation with zero-shot
    args.gender='male'
    args.pitch='moderate'
    args.speed='moderate'
    args.save_dir = 'example/seedtts-results-male_moderate_moderate'
    
    main(args)
    