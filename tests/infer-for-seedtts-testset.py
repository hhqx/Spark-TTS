"""
This is a script to perform TTS inference on the SeedTTS test dataset (linux only).
It uses multiprocessing to handle multiple GPUs and saves the generated audio files.

usage:

# test wer test set
    python tests/infer-for-seedtts-testset.py --dataset_split test_wer --save_dir outputs/seedtts/wer

# test sim test set
    python tests/infer-for-seedtts-testset.py --dataset_split test_sim --save_dir outputs/seedtts/sim

"""

import os
import argparse
import torch
import soundfile as sf
import logging
from datetime import datetime
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


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run TTS inference.")

    parser.add_argument(
        "--model_dir",
        type=str,
        default="pretrained_models/Spark-TTS-0.5B",
        help="Path to the model directory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/seedtts/wer",
        help="Directory to save generated audio files",
    )
    parser.add_argument("--cuda_visible_devices", type=str, default=None, help="CUDA visible devices, e.g., '0,1,2'")
    parser.add_argument(
        "--dataset_path", default='hhqx/seedtts_testset', type=str, required=False, help="Path to the seedtts test dataset"
    )
    parser.add_argument("--dataset_lang_config", choices=["en", "zh"], default="en", help="Language of the test dataset")
    parser.add_argument(
        "--dataset_split",
        choices=['test_wer', 'test_sim'],
        default='test_wer',
        help="Split of the dataset to use for inference",
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
            item['audio_output_path'] = os.path.abspath(item['audio_output_path'])
            
            save_path = os.path.join(out_wav_dir, output_name)
            save_path = os.path.abspath(save_path)
            
            # res = {
            #     'index': item.get('index', f"rank{rank}_{item_count}"),
            #     'text': text,
            #     'wav_path': save_path,
            #     'ref_wav_path': prompt_speech_path,
            #     'prompt_text': prompt_text
            # }
            res = item.copy()
            
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
                        import traceback
                        traceback.print_exc()
                        logging.error(f"Rank {rank} - Error during inference for text '{text}': {str(e)}", exc_info=True)
                        wav = None
                    
                    if wav is None:
                        res['wav_path'] = ''
                        res['audio_output_path'] = ''
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
    dataset = load_dataset(
        config.dataset_path,
        **{"name": config.dataset_lang_config, "split": config.dataset_split}
        )
    
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
    
    # save json file
    with open(os.path.join(output_dir, f'text_wav_{os.path.basename(output_dir)}.json'), 'w') as f:
        json.dump(text_wav_pairs, f, ensure_ascii=False, indent=4)
    
    logging.info(f"Successfully processed {len(text_wav_pairs)} items")
    logging.info(f"Results saved to {output_dir}")
    
    print("Text and wav pairs saved to (len={}, error_wav={}): {} \n".format(
            len(text_wav_pairs),
            sum(1 for item in text_wav_pairs if not item['audio_output_path']),
            os.path.join(output_dir, f'text_wav_{os.path.basename(output_dir)}.json')
        ))

    # save .lst file for seedtts evaluation
    lst_file_path = os.path.join(
        output_dir, 
        'result_{}_{}_model_{}.lst'.format(
            config.dataset_lang_config, config.dataset_split, os.path.basename(config.model_dir)
        )
    )
    
    with open(lst_file_path, 'w') as f:
        # dataset[0]: dict_keys(['audio_output_path', 'prompt_text', 'prompt_audio', 'text_input', 'audio_ground_truth'])
        for item in text_wav_pairs:
            line = "|".join([
                os.path.splitext(os.path.basename(item['audio_output_path']))[0],
                item['prompt_text'],
                item['prompt_audio'],
                item['text_input'],
                item['audio_ground_truth']
            ]) + "\n"
            f.write(line)
    logging.info(f"Successfully saved .lst file to (len={len(text_wav_pairs)}): {lst_file_path}")
    
    print('\n\n-------------- Commands to do seedtts tests using generated wavs -----------------------------')
    print(f"Successfully saved .lst file to (len={len(text_wav_pairs)}): {lst_file_path}")
    seed_repo_url = 'https://github.com/BytedanceSpeech/seed-tts-eval'
    synth_wav_dir = os.path.join(os.path.abspath(config.save_dir), 'wav')
    lst_file_path = os.path.abspath(lst_file_path)
    tpl = 'Use the following command (from "{seed_repo_url}") to evaluate the results:'.format(seed_repo_url=seed_repo_url)
    cmd1 = 'bash cal_wer.sh {the path of .lst meta file} {the directory of synthesized audio} en'
    cmd2 = 'bash cal_sim.sh {the path of .lst meta file} {the directory of synthesized audio} {path/wavlm_large_finetune.pth}'
    print(tpl)
    print(f'''```bash
# For WER evaluation:
# bash cal_wer.sh {{the path of .lst meta file}} {{the directory of synthesized audio}} en
bash cal_wer.sh {lst_file_path} {synth_wav_dir} en

# For SIM evaluation:
# bash cal_sim.sh {{the path of .lst meta file}} {{the directory of synthesized audio}} {{path/wavlm_large_finetune.pth}}
bash cal_sim.sh {lst_file_path} {synth_wav_dir} {{'path/wavlm_large_finetune.pth'}}
```''')
    print('--------------------------------------------------------------\n\n')


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    args = parse_args()
    
    if not os.path.exists(args.model_dir):
        raise ValueError(f"Model directory does not exist: {args.model_dir}, please check the pretrained spark tts model path.")
    
    cuda_visible_devices = args.cuda_visible_devices
    if cuda_visible_devices is None:
        print("No CUDA_VISIBLE_DEVICES specified,"
              " using default CUDA_VISIBLE_DEVICES={}".format(os.environ.get("CUDA_VISIBLE_DEVICES", "")))
    else:
        # Set CUDA_VISIBLE_DEVICES environment variable
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
        print(f"Using CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
    
    # switch here to test generation with zero-shot
    # args.gender='male'
    # args.pitch='moderate'
    # args.speed='moderate'
    
    # generate result for WER test set
    # args.save_dir = 'outputs/seedtts/wer'
    # args.dataset_split = 'test_wer'
    
    # # # # generate result for SIM test set
    # args.save_dir = 'outputs/seedtts/sim'
    # args.dataset_split = 'test_sim'

    main(args)
    