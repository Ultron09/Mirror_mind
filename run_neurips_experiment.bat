@echo off
title AirborneHRS NeurIPS Benchmark (RTX 3050 Optimized)

echo [SETUP] Activating Conda Environment: torch-gpu...
call activate torch-gpu
if %ERRORLEVEL% NEQ 0 (
    echo Warning: 'call activate' failed. Trying 'conda activate'...
    call conda activate torch-gpu
)

echo [SETUP] Configuring CUDA Memory...
:: Helps prevent memory fragmentation errors on 6GB RAM
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo ====================================================
echo      AirborneHRS NeurIPS Scientific Benchmark
echo ====================================================
echo.
echo This experiment runs 3 seeds (42, 43, 44) for 100 Epochs.
echo Safe Mode: Batch Size limited to 64 for RTX 3050.
echo.
echo Press Ctrl+C to stop. You can resume later by running this again.
echo.
pause

:start_benchmark
echo [%date% %time%] Starting Benchmark Script...
python benchmark_continual.py --seeds 42 43 44 --epochs 100
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] Script crashed or interrupted.
    echo restarting in 10 seconds...
    timeout /t 10
    goto start_benchmark
)

echo.
echo [%date% %time%] Benchmark Complete. Generating Plots...
python plot_benchmark.py

echo.
echo ====================================================
echo      Experiment Finished Successfully
echo ====================================================
pause
