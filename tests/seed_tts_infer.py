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

from cli.SparkTTS import SparkTTS


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




import torch.multiprocessing as mp
from tqdm import tqdm
from typing import List, Dict

# def run_tts(args):
def process_rank(rank, args, data: List[dict], shared_results=None, lock=None):
    """Perform TTS inference and save the generated audio."""
    logging.info(f"Using model from: {args.model_dir}")
    logging.info(f"Saving audio to: {args.save_dir}")

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Convert device argument to torch.device
    device = torch.device(f'cuda:{rank}')
    
    # Initialize the model
    model = SparkTTS(args.model_dir, device)

    # Generate unique filename using timestamp
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(args.save_dir, f"{timestamp}.wav")

    logging.info("Starting inference...")
    
    args.ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if args.ngpu < 1:
        raise ValueError("Number of GPUs (ngpu) must be at least 1.")
    data = data[rank::args.ngpu]

    text_wav_pairs = []
    for i, item in enumerate(tqdm(data, desc=f"Rank {rank} processing", unit="item")):
        
        text = item['text_input']
        prompt_text = item['prompt_text']
        prompt_speech_path = item['prompt_audio']
        output_name = item['audio_output_path']

        out_wav_dir = os.path.join(args.save_dir, 'wav')
        os.makedirs(out_wav_dir, exist_ok=True)
        item['audio_output_path'] = os.path.join(out_wav_dir, output_name)
        save_path = os.path.join(out_wav_dir, output_name)
        
        res = {
            'index': item.get('index', "rank{}_{}".format(rank, i)),
            'text': text,
            'wav_path': os.path.join(out_wav_dir, output_name),
            'ref_wav_path': prompt_speech_path,
            'prompt_text': prompt_text
            }
        
        if os.path.exists(save_path):
            logging.info(f"File already exists: {save_path}, skipping inference.")
        else:
            # Perform inference and save the output audio
            with torch.no_grad():
                wav = model.inference(
                    text,
                    prompt_speech_path,
                    prompt_text=prompt_text,
                    gender=args.gender,
                    pitch=args.pitch,
                    speed=args.speed,
                )
                if wav is None:
                    res['wav_path'] = ''
                    logging.warning(f"Skipping empty output for text: {text}")
                else:
                    sf.write(save_path, wav, samplerate=16000)
                    logging.info(f"Audio saved at: {save_path}")
        text_wav_pairs.append(res)
    
    # text_wav_pairs = data
    
    if lock is not None:
        # 写入共享内存（关键步骤：使用锁确保线程安全）
        with lock:  # 加锁避免多进程同时写入导致的数据错乱
            shared_results.extend(text_wav_pairs)
    else:
        # 如果不是多进程，直接返回结果
        print(f"Rank {rank} processed {len(text_wav_pairs)} samples.")
        return text_wav_pairs


from torch.multiprocessing import Manager
import os
from tqdm import tqdm

# 确保多进程启动方式正确
mp.set_start_method('spawn', force=True)

from datasets import load_dataset
import json

def main(config):
    ## load dataset
    input_dataset_path = '/hy-netdisk/datasets_processed/seedtts_testset'
    dataset = load_dataset(input_dataset_path, **{"name": "en", "split": "test_wer"})
    # dataset = dataset.to_pandas().to_dict(orient='records')
    dataset.to_pandas().to_dict(orient='records', index=True)  # Convert to list of dicts
    df = dataset.to_pandas()
    df = df.reset_index()  # 关键：索引会被添加为名为 'index' 的列
    dataset = df.to_dict(orient='records')
    
    ## process dataset
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if world_size < 1:
        raise ValueError("Number of GPUs (ngpu) must be at least 1.")
    
    # ipdb.set_trace()  # Debugging breakpoint
    if world_size == 1:
        # If only one GPU, run directly in the main process
        print("Running on a single GPU, no multiprocessing needed.")
        text_wav_pairs = process_rank(0, config, dataset)
    else:
        # 创建共享内存容器（由Manager管理，支持跨进程访问）
        with Manager() as manager:
            # 共享列表：存储所有进程的结果
            shared_results = manager.list()
            # 共享锁：确保多进程写入时的数据安全
            lock = manager.Lock()
            
            # 启动多进程
            mp.spawn(
                process_rank,
                args=(config, dataset, shared_results, lock),
                nprocs=world_size,
                join=True
            )
            
            # 主进程直接从共享列表获取结果（已自动合并）
            text_wav_pairs = list(shared_results)  # 转换为普通列表便于后续处理
            print(f"Total results collected: {len(text_wav_pairs)}")
    
    text_wav_pairs.sort(key=lambda x: x['index'])
    
    output_dir = config.save_dir
    
    with open(os.path.join(output_dir, f'text_wav_{os.path.basename(output_dir)}.json'), 'w') as f:
        json.dump(text_wav_pairs, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    main(args)
    # run_tts(args)
