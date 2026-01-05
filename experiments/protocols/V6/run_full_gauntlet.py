
"""
PROTOCOL V6: THE FULL TITAN GAUNTLET (MASTER ORCHESTRATOR)
==========================================================
Executes all implemented phases of the Titan Protocol (V1 Port) and generates a unified report.

Phases:
1. Integrity (Drifting Sinusoid)
2. Verification (Split-CIFAR10 Memory)
3. Universal (Mackey-Glass Stability)
4. Behavior (Intrusion Defense)
7. SOTA Deathmatch (Drone Sim)
"""

import subprocess
import sys
import os
import time
import logging

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("TitanGauntlet")

PHASES = [
    ("Phase 1: Integrity (Plasticity)", "phases/phase1_integrity.py"),
    ("Phase 2: Verification (Memory)", "phases/phase2_memory.py"),
    ("Phase 3: Universal (Stability)", "phases/phase3_stability.py"),
    ("Phase 4: Behavior (Defense)", "phases/phase4_defense.py"),
    ("Phase 7: SOTA Deathmatch (Drone)", "phases/phase7_deathmatch.py"),
]

def run_gauntlet():
    logger.info("⚔️  INITIATING TITAN GAUNTLET (PROTOCOL V6) ⚔️")
    logger.info("===============================================")
    
    results = []
    
    for name, script in PHASES:
        logger.info(f"\n\n>>> STARTING {name} <<<")
        start_time = time.time()
        
        script_path = os.path.join(os.path.dirname(__file__), script)
        
        try:
            # Run as subprocess to ensure clean state
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                check=False # Don't crash master on sub-failure
            )
            
            duration = time.time() - start_time
            status = "PASSED" if result.returncode == 0 else "FAILED"
            
            # Check logs for specific success keywords if returncode is 0
            if status == "PASSED":
                if "FAILED" in result.stdout or "WARNING" in result.stdout:
                    status = "WARNING"
                if "PASSED" in result.stdout:
                    status = "PASSED"
            
            logger.info(f"   Result: {status} ({duration:.2f}s)")
            
            # Print last few lines of output
            output_lines = result.stdout.strip().split('\n')
            logger.info("   Output Tail:")
            for line in output_lines[-5:]:
                logger.info(f"      {line}")
                
            if result.stderr:
                logger.warning("   Errors:")
                for line in result.stderr.strip().split('\n')[-5:]:
                    logger.warning(f"      {line}")
            
            results.append({
                'name': name,
                'status': status,
                'duration': duration,
                'output': result.stdout
            })
            
        except Exception as e:
            logger.error(f"   ❌ EXECUTION ERROR: {e}")
            results.append({
                'name': name,
                'status': "ERROR",
                'duration': 0,
                'output': str(e)
            })

    # Generate Report
    generate_report(results)

def generate_report(results):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Protocol V6: Titan Gauntlet Results
**Date:** {timestamp}

## Summary
| Phase | Status | Duration |
| :--- | :--- | :--- |
"""
    
    for r in results:
        icon = "✅" if r['status'] == "PASSED" else ("⚠️" if r['status'] == "WARNING" else "❌")
        report += f"| {r['name']} | {icon} {r['status']} | {r['duration']:.2f}s |\n"
        
    report += "\n## Detailed Logs\n"
    
    for r in results:
        report += f"\n### {r['name']}\n"
        report += "```\n"
        report += r['output'][-2000:] # Last 2000 chars
        report += "\n```\n"
        
    with open(os.path.join(os.path.dirname(__file__), "TITAN_GAUNTLET_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
        
    logger.info("\n\n✅ GAUNTLET COMPLETE. Report saved to TITAN_GAUNTLET_REPORT.md")

if __name__ == "__main__":
    run_gauntlet()
