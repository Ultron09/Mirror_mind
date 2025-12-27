#!/usr/bin/env python3
"""Get core.py statistics"""
with open('airbornehrs/core.py', 'rb') as f:
    content = f.read().decode('utf-8', errors='ignore')
    lines = content.split('\n')
    
print('=== CORE.PY STATISTICS ===\n')
print(f'Total Lines: {len(lines)}')
print(f'Non-empty Lines: {len([l for l in lines if l.strip()])}')
print(f'Classes: {len([l for l in lines if l.strip().startswith("class ")])}')
print(f'Functions: {len([l for l in lines if l.strip().startswith("def ")])}')

# Count imports
imports = len([l for l in lines if l.strip().startswith('import ') or l.strip().startswith('from ')])
print(f'Imports: {imports}')

# Count comments  
comments = len([l for l in lines if '#' in l])
print(f'Lines with comments: {comments}')
