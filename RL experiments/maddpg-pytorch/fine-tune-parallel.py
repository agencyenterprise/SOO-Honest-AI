import subprocess
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
import time
import sys
import os

def get_stage_label(stage):
    if stage == 0:
        return 'pretraining'
    elif stage == 1:
        return 'fine-tuning without overlap'
    elif stage == 2:
        return 'fine-tuning with overlap'

def get_latest_run_number(model_dir):
    runs = [d for d in os.listdir(model_dir) if d.startswith('run')]
    if not runs:
        return 0
    latest_run = max(runs, key=lambda x: int(x[3:]))
    return int(latest_run[3:])

def run_subprocess(command):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='')
            sys.stdout.flush()
    return process

def run_fine_tune(args, repetitions, initial_parallel, seeds):
    model_dir = f"./models/{args.env_id}/{args.model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    seed_run_numbers = {}
    stage_1_runs=[]
    stage_2_runs=[]

    with ThreadPoolExecutor(max_workers=initial_parallel) as executor:
            for stage in range(3):
                for i in range(initial_parallel):
                    futures = []
                    if seeds[i] not in seed_run_numbers:
                        seed_run_numbers[seeds[i]] = []

                    time.sleep(2)

                    future = executor.submit(run_stages, args, seeds[i], seed_run_numbers, model_dir, stage)
                    futures.append(future)

                    for future in futures:
                        seed, seed_run_numbers = future.result()

    for i in range(initial_parallel, repetitions):
        for stage in range(3):
            if seeds[i] not in seed_run_numbers:
                    seed_run_numbers[seeds[i]] = []

            seed, seed_run_numbers = run_stages(args, seeds[i], seed_run_numbers, model_dir, stage)

    print("Seed    | Stage 0 | Stage 1 | Stage 2")
    print("------------------------------------")
    for seed, run_numbers in seed_run_numbers.items():
        print(f"{seed:<7} | {run_numbers[0]:<7} | {run_numbers[1]:<7} | {run_numbers[2]:<7}")
        stage_1_runs.append(run_numbers[1])
        stage_2_runs.append(run_numbers[2])

    with_command = ' '.join(map(str, stage_1_runs))
    without_command = ' '.join(map(str, stage_2_runs))
    plot_command = f"python multiple_seed_plot_means.py --with {with_command} --without {without_command} --window 1500 --type 'with Self-Other Overlap'"
    print(f"Executing plot command: {plot_command}")
    subprocess.run(plot_command, shell=True)

def run_stages(args, seed, seed_run_numbers, model_dir, stage):
    load_run_number = seed_run_numbers[seed][0] if stage > 0 else None
    command = build_command(args, seed, stage, load_run_number)
    print(f'Running {get_stage_label(stage)} for seed {seed}')
    print(f"Executing command: {' '.join(command)}")
    seed_run_numbers[seed].append(get_latest_run_number(model_dir) + 1)
    run_subprocess(command)
    return seed, seed_run_numbers

def build_command(args, seed, stage, load_run_number=None):
    command = ["python", "-u", "fine-tune.py", args.env_id, args.model_name]

    if stage > 0:
        command.append("--load")
        command.append(str(load_run_number))
    if stage == 2:
        command.append("--self_other")
        command.append("True")

    for arg, value in vars(args).items():
        if arg not in ["n_reps", "n_init", "seeds", "env_id", "model_name", "self_other", "load"]:
            if arg == "pre_trained" and value is None:
                continue
            if arg == "year" and value is None:
                continue
            if isinstance(value, bool):
                if value:
                    command.append(f"--{arg}")
            else:
                command.append(f"--{arg}")
                command.append(str(value))
    command.append("--seed")
    command.append(str(seed))
    return command

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_reps", type=int, default=1, help="Total number of repetitions to run fine-tune.py")
    parser.add_argument("--n_init", type=int, default=1, help="Number of initial parallel runs")
    parser.add_argument("--seeds", nargs="*", type=int, help="List of seeds for each repetition")

    # arguments from fine-tune.py
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name", help="Name of directory to store model/training contents")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    # parser.add_argument("--load", type=int, default=None, help="Optional load argument")
    parser.add_argument("--pre_trained", type=int, default=None, help="Optional pre-trained argument")
    parser.add_argument("--self_other", default=False, type=bool)
    parser.add_argument("--random", default=False, type=bool)
    parser.add_argument("--eval", default=False, type=bool)
    parser.add_argument("--year", type=int, default=None, help="Optional pre-trained argument")
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=25000, type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size", default=1024, type=int, help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=25000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action", action='store_true')

    args = parser.parse_args()

    if args.seeds is None:
        args.seeds = [random.randint(1, 10000) for _ in range(args.n_reps)]
    elif len(args.seeds) != args.n_reps:
        raise ValueError("The length of the seeds array must match the total number of repetitions.")

    run_fine_tune(args, args.n_reps, args.n_init, args.seeds)