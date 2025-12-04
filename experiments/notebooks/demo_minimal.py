# Minimal demo script (non-notebook) that runs the sample experiment and prints a short summary

if __name__ == '__main__':
    from experiments.run_experiment import run
    out = run('experiments/configs/sample_config.json', output_dir='runs/demo')
    print('Experiment summary:')
    print(out)
