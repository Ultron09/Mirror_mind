@echo off
title AirborneHRS NeurIPS Benchmark Experiment
echo ====================================================
echo      AirborneHRS NeurIPS Scientific Benchmark
echo ====================================================
echo.
echo This experiment runs 3 seeds (42, 43, 44) for 100 Epochs.
echo ESTIMATED RUNTIME: ~90 Hours on single GPU.
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
