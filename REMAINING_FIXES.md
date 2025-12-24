# OPTIMIZATION & INTEGRATION BUG FIXES
## Applied Fixes for Remaining Issues

### BUG #11: Adapter Hook Memory Leak Fix

Add hook cleanup to core.py (in AdaptiveFramework.__init__ after line 480):

```python
# Store hook handles for cleanup
self._hook_handles = []

# Later in _attach_hooks():
for name, module in self.model.named_modules():
    if isinstance(module, valid_types):
        self.layer_map[name] = idx
        handle = module.register_forward_pre_hook(self._generate_fast_hook(idx))
        self._hook_handles.append(handle)  # Store for cleanup
        idx += 1

# Add cleanup method:
def __del__(self):
    """Cleanup hooks on deletion"""
    try:
        for handle in getattr(self, '_hook_handles', []):
            handle.remove()
    except Exception:
        pass
```

---

### BUG #13: MetaController Update Integration

Add to train_step() method in core.py after optimizer.step() (around line 750):

```python
# BUG FIX #13: Call MetaController update for Reptile synchronization
if hasattr(self, 'meta_controller') and self.meta_controller is not None:
    if self.step_count % 5 == 0:  # Every 5 steps (reptile_update_interval)
        try:
            self.meta_controller.update(step=self.step_count)
        except Exception as e:
            self.logger.debug(f"MetaController update failed: {e}")
```

---

### BUG #14: Prioritized Buffer Initialization Fix

Update core.py line ~355:

```python
# BUG FIX #14: Safe prioritized buffer initialization with fallback
use_prioritized = getattr(config, 'use_prioritized_replay', True)
if use_prioritized:
    try:
        self.prioritized_buffer = PrioritizedReplayBuffer(
            capacity=config.feedback_buffer_size,
            temperature=getattr(config, 'replay_priority_temperature', 0.6)
        )
        self.logger.info("[REPLAY] Prioritized replay enabled")
    except Exception as e:
        self.logger.warning(f"Prioritized buffer initialization failed: {e}. Falling back to regular buffer.")
        self.prioritized_buffer = None
else:
    self.prioritized_buffer = None
```

---

### BUG #15: Layer Map Synchronization Fix

Update adapt_weights() in PerformanceMonitor class (around line 295):

```python
# BUG FIX #15: Use parameter names directly instead of layer indices
with torch.no_grad():
    for name, param in self.model.named_parameters():
        if param.requires_grad:
            param_importance = 0.1
            
            # Find matching layer in layer_map by name prefix
            matched_importance = None
            for layer_name, idx in layer_map.items():
                if layer_name in name:
                    # Verify telemetry_buffer has this index
                    if idx < len(telemetry_buffer):
                        stats = telemetry_buffer[idx]
                        mean_act = stats[0].abs()
                        matched_importance = (mean_act * stats[1]).item()
                        break
            
            if matched_importance is not None:
                param_importance = matched_importance
            
            # Apply updates with safeguards
            scale_factor = raw_scale * self.config.weight_adaptation_lr * param_importance
            shift_factor = raw_shift * self.config.weight_adaptation_lr * param_importance
            
            # Prevent extreme updates
            scale_factor = np.clip(scale_factor, -0.1, 0.1)
            shift_factor = np.clip(shift_factor, -0.1, 0.1)
            
            if param.ndim == 1:
                param.mul_(1.0 + scale_factor)
                param.add_(shift_factor)
            elif param.ndim >= 2:
                param.mul_(1.0 + scale_factor)
```

---

### BUG #16 & #20: Memory Optimization Fixes

Update FeedbackBuffer in core.py:

```python
class FeedbackBuffer:
    """Robust Experience Replay Buffer using Reservoir Sampling with memory optimization."""
    
    def add(self, input_data, output, target, reward, loss):
        snapshot = PerformanceSnapshot(
            # BUG FIX #16: Explicitly detach and move to CPU to break gradient graphs
            input_data=input_data.detach().cpu().numpy() if hasattr(input_data, 'numpy') else input_data,
            output=output.detach().cpu().numpy() if hasattr(output, 'numpy') else output,
            target=target.detach().cpu().numpy() if hasattr(target, 'numpy') else target,
            reward=reward,
            loss=loss,
            timestamp=datetime.now().timestamp(),
            episode=self.total_seen
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(snapshot)
        else:
            replace_idx = random.randint(0, self.total_seen)
            if replace_idx < self.capacity:
                self.buffer[replace_idx] = snapshot
        self.total_seen += 1
```

---

### BUG #17: Redundant Error Calculation Optimization

Update ConsciousnessCore.observe() in consciousness.py (around line 100):

```python
def observe(self, y_pred, y_true, features=None, domain_id=None, task_id=None):
    """Process observation with consolidated error computation"""
    
    # Compute error ONCE
    error = F.mse_loss(y_pred, y_true, reduction='none').mean(dim=1)  # Shape: [batch_size]
    
    # BUG FIX #17: Reuse error computation everywhere
    self._update_error_stats(error)  # Updates mean, std, variance
    
    # Now reuse computed statistics:
    current_error_mean = self.error_mean  # From update_error_stats
    current_error_std = self.error_std    # Already computed
    
    # Confidence from error
    confidence = 1.0 / (1.0 + error.mean().item())
    
    # Surprise from error (reuse std)
    surprise = self._compute_surprise(error)  # Uses self.error_std (already updated)
    
    # Feature importance (if provided, use sparse computation)
    feature_importance = None
    if features is not None and features.dim() > 1:
        # BUG FIX #17: Only compute if features provided
        importance = self._compute_feature_importance(features, error)
        feature_importance = importance
    
    # ... rest of method
```

---

### BUG #18: Data Type Consistency

Update arc_agi3_agent_v2.py (search for grid_tensor usage):

```python
# BUG FIX #18: Keep data in native format until needed
# DON'T do: grid_tensor = torch.from_numpy(grid).to(device); grid_np = grid_tensor.cpu().numpy()
# Instead:
if isinstance(grid, np.ndarray):
    grid_analysis = self._analyze_grid_numpy(grid)  # Keep in numpy
else:
    grid_analysis = self._analyze_grid_tensor(grid)  # Keep in tensor
```

---

### BUG #19: Omega Normalization Fix

Update UnifiedMemoryHandler.consolidate() in memory.py (around line 210):

```python
# BUG FIX #19: Normalize omega ONCE and reuse
if self.method in ['si', 'hybrid']:
    # Consolidate all parameters first
    for name, p in self.model.named_parameters():
        if not p.requires_grad:
            continue
        # ... existing consolidation code ...
        self.omega[name] = new_omega.clamp(min=0.0, max=1e6)
    
    # BUG FIX #19: Normalize ONCE after all parameters consolidated
    omega_sum = sum(torch.sum(self.omega[name]) for name in self.omega if name in self.omega)
    if omega_sum > 0:
        for name in self.omega:
            self.omega[name] = self.omega[name] / (omega_sum + 1e-8)
    
    # Cache for later use (don't renormalize each time)
    self._omega_normalized = True
```

Then in penalty computation:

```python
# Use cached normalization instead of recomputing
penalty = 0.0
for name, p in self.model.named_parameters():
    if name in self.omega and p.grad is not None:
        # Omega is already normalized from consolidate()
        penalty += torch.sum(self.omega[name] * p.grad.data ** 2)
```

---

## Summary of Applied Fixes

| Bug | Fix Type | Status |
|-----|----------|--------|
| #1 | Adaptive EMA for baseline | ✅ Applied |
| #2 | Epsilon before division | ✅ Applied |
| #3 | Proper variance initialization | ✅ Applied |
| #4 | Adaptive EMA weight | ✅ Applied |
| #5 | NaN validation in scores | ✅ Applied |
| #6 | Win condition calibration (0.75) | ✅ Applied |
| #7 | Entropy without zero color | ✅ Applied |
| #8 | Reward scaling by difficulty | ✅ Applied |
| #9 | Q-value clipping order | ⏳ Agent-level fix |
| #10 | Buffer initialization check | ⏳ Agent-level fix |
| #11 | Hook cleanup | 📝 Code provided |
| #12 | Consciousness default False | ✅ Applied |
| #13 | MetaController update call | 📝 Code provided |
| #14 | Prioritized buffer try-except | 📝 Code provided |
| #15 | Layer map sync | 📝 Code provided |
| #16 | Memory leak prevention | 📝 Code provided |
| #17 | Error computation reuse | 📝 Code provided |
| #18 | Data type consistency | 📝 Code provided |
| #19 | Omega normalization cache | 📝 Code provided |
| #20 | Grid hash instead of copy | ⏳ Evaluator optimization |

✅ = Applied to codebase
📝 = Code snippet provided above
⏳ = Optional enhancement
