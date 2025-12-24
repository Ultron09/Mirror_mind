"""
Comprehensive Example: MirrorMind Self-Awareness in Action
============================================================

This example demonstrates the full power of the human-like self-awareness
framework applied to a multi-domain learning scenario.

We'll train on:
1. Vision domain (MNIST-like classification)
2. Language domain (text classification)
3. Audio domain (signal processing)

The self-aware framework will:
- Automatically detect learning phases
- Adjust learning rates per domain
- Identify and prioritize weak areas
- Plan learning trajectory
- Estimate time to mastery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Tuple

# Import the self-awareness framework
from airbornehrs import (
    HumanLikeSelfAwarenessWrapper,
    MirrorMindWithSelfAwareness,
    AdaptiveLearningController
)


# ============================================================================
# MULTI-DOMAIN MODEL
# ============================================================================

class MultiDomainModel(nn.Module):
    """
    Model with separate branches for vision, language, and audio domains.
    """
    
    def __init__(self):
        super().__init__()
        
        # Vision branch: Image classification
        self.vision_encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.vision_head = nn.Linear(128, 10)
        
        # Language branch: Text classification
        self.language_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        self.language_head = nn.Linear(128, 5)
        
        # Audio branch: Audio classification
        self.audio_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.audio_head = nn.Linear(64, 8)
        
        # Shared processing
        self.shared = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x: torch.Tensor, domain: str = 'vision') -> torch.Tensor:
        """
        Forward pass for specified domain.
        """
        if domain == 'vision':
            encoded = self.vision_encoder(x)
            output = self.vision_head(encoded)
        elif domain == 'language':
            encoded = self.language_encoder(x)
            output = self.language_head(encoded)
        elif domain == 'audio':
            encoded = self.audio_encoder(x)
            output = self.audio_head(encoded)
        else:
            raise ValueError(f"Unknown domain: {domain}")
        
        return output


# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_multi_domain_data(num_samples: int = 1000) -> Dict[str, Tuple]:
    """
    Generate synthetic multi-domain data.
    """
    
    # Vision data: (samples, 784) -> 10 classes
    vision_x = torch.randn(num_samples, 784)
    vision_y = torch.randint(0, 10, (num_samples,))
    
    # Language data: (samples, 512) -> 5 classes
    language_x = torch.randn(num_samples, 512)
    language_y = torch.randint(0, 5, (num_samples,))
    
    # Audio data: (samples, 256) -> 8 classes
    audio_x = torch.randn(num_samples, 256)
    audio_y = torch.randint(0, 8, (num_samples,))
    
    return {
        'vision': (vision_x, vision_y),
        'language': (language_x, language_y),
        'audio': (audio_x, audio_y)
    }


# ============================================================================
# TRAINING WITH SELF-AWARENESS
# ============================================================================

def train_with_self_awareness(num_steps: int = 5000, report_interval: int = 500):
    """
    Train the model with self-awareness framework.
    
    The model will:
    1. Dynamically adjust learning rates per domain
    2. Identify which domains need more learning
    3. Plan its learning trajectory
    4. Track learning phases
    """
    
    print("="*70)
    print("TRAINING WITH HUMAN-LIKE SELF-AWARENESS")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultiDomainModel().to(device)
    base_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion_vision = nn.CrossEntropyLoss()
    criterion_language = nn.CrossEntropyLoss()
    criterion_audio = nn.CrossEntropyLoss()
    
    # Wrap with self-awareness
    aware_model = MirrorMindWithSelfAwareness(model, buffer_size=5000)
    
    # Generate data
    data = generate_multi_domain_data(num_samples=1000)
    domains = ['vision', 'language', 'audio']
    domain_loaders = {}
    
    for domain in domains:
        x, y = data[domain]
        dataset = TensorDataset(x, y)
        domain_loaders[domain] = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    step_count = 0
    domain_cycle = 0
    
    while step_count < num_steps:
        # Cycle through domains
        domain = domains[domain_cycle % 3]
        loader = domain_loaders[domain]
        
        # Select criterion
        if domain == 'vision':
            criterion = criterion_vision
        elif domain == 'language':
            criterion = criterion_language
        else:
            criterion = criterion_audio
        
        for batch_x, batch_y in loader:
            if step_count >= num_steps:
                break
            
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Forward pass
            output = model(batch_x, domain=domain)
            loss = criterion(output, batch_y)
            
            # Update self-awareness
            confidence_signal = aware_model.observe(
                output, batch_y.unsqueeze(1).float() / 10.0,  # Normalize for awareness
                domain_id=domain,
                input_data=batch_x
            )
            
            # Get adaptive learning rate
            adaptive_lr = aware_model.get_adaptive_lr(domain, base_lr=1e-3)
            for param_group in base_optimizer.param_groups:
                param_group['lr'] = adaptive_lr
            
            # Backward pass
            base_optimizer.zero_grad()
            loss.backward()
            base_optimizer.step()
            
            step_count += 1
            
            # Periodic reporting
            if step_count % report_interval == 0:
                print(f"\n{'='*70}")
                print(f"Step {step_count}/{num_steps}")
                print(f"{'='*70}")
                
                # Get awareness state
                state = aware_model.get_awareness_state()
                insights = aware_model.get_awareness_insights()
                
                # Print current status
                print(f"\n📊 CURRENT STATUS:")
                print(f"  Phase: {state.phase.name}")
                print(f"  Global Confidence: {state.global_confidence:.1%}")
                print(f"  Global Competence: {state.global_competence:.1%}")
                print(f"  Global Uncertainty: {state.global_uncertainty:.4f}")
                
                print(f"\n🎯 DOMAIN COMPETENCE:")
                for domain_name, competence in state.confidence_by_domain.items():
                    bar = '█' * int(competence * 20) + '░' * (20 - int(competence * 20))
                    print(f"  {domain_name:12s} {bar} {competence:.1%}")
                
                print(f"\n🔍 LEARNING DIRECTION:")
                print(f"  {state.learning_direction}")
                
                print(f"\n⚡ IMPROVEMENT PRIORITIES:")
                for i, priority in enumerate(state.prioritized_improvements, 1):
                    print(f"  {i}. {priority}")
                
                print(f"\n⚙️  CURRENT BOTTLENECKS:")
                for bottleneck in state.current_bottlenecks:
                    print(f"  • {bottleneck}")
                
                print(f"\n📈 LEARNING RECOMMENDATIONS:")
                print(f"  Learning Rate Multiplier: {insights['learning_rate_adjustment']:.2f}x")
                print(f"  Exploration Ratio: {insights['exploration_ratio']:.1%}")
                print(f"  Current Loss: {loss.item():.4f}")
                
                # Get learning plan
                plan = aware_model.get_improvement_plan(horizon=num_steps - step_count)
                print(f"\n🗺️  LEARNING PLAN:")
                print(f"  Primary Focus: {plan['primary_focus']}")
                print(f"  Secondary Focuses: {plan['secondary_focuses']}")
                if plan['estimated_milestones']:
                    print(f"  Next Milestone: {plan['estimated_milestones'][0]}")
                
                if plan['transfer_learning_opportunities']:
                    print(f"\n🔗 TRANSFER LEARNING OPPORTUNITIES:")
                    for src, dst in plan['transfer_learning_opportunities']:
                        print(f"  {src} → {dst}")
        
        domain_cycle += 1
    
    # Final comprehensive report
    print("\n" + "="*70)
    print("FINAL SELF-AWARENESS REPORT")
    print("="*70)
    aware_model.print_report()
    
    return aware_model, model


# ============================================================================
# ANALYSIS: DEMONSTRATE SELF-AWARENESS CAPABILITIES
# ============================================================================

def analyze_self_awareness(aware_model: MirrorMindWithSelfAwareness):
    """
    Deep dive into what the self-aware model learned about itself.
    """
    
    print("\n" + "="*70)
    print("SELF-AWARENESS ANALYSIS")
    print("="*70)
    
    state = aware_model.get_awareness_state()
    
    # 1. Learning Phase Analysis
    print(f"\n1️⃣  LEARNING PHASE ANALYSIS")
    print(f"   Current Phase: {state.phase.name}")
    print(f"   ")
    print(f"   What this means:")
    if state.phase.name == 'EXPLORATION':
        print(f"   → Still discovering new concepts")
        print(f"   → High learning rate recommended")
        print(f"   → Seek diverse examples")
    elif state.phase.name == 'CONSOLIDATION':
        print(f"   → Stabilizing knowledge")
        print(f"   → Moderate learning rate")
        print(f"   → Focus on hard examples")
    elif state.phase.name == 'MASTERY':
        print(f"   → Approaching expert level")
        print(f"   → Low learning rate (fine-tuning)")
        print(f"   → Optimize performance edge cases")
    
    # 2. Competence by Domain
    print(f"\n2️⃣  DOMAIN COMPETENCE RANKING")
    sorted_domains = sorted(
        state.confidence_by_domain.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for rank, (domain, competence) in enumerate(sorted_domains, 1):
        stars = '⭐' * int(competence * 5)
        print(f"   #{rank} {domain:15s} {competence:.1%} {stars}")
    
    # 3. Capability Gaps
    print(f"\n3️⃣  CAPABILITY GAPS (What needs improvement?)")
    for gap_name, importance in state.capability_gaps[:5]:
        importance_pct = importance * 100
        bar = '█' * int(importance_pct // 5) + '░' * (20 - int(importance_pct // 5))
        print(f"   {gap_name:20s} {bar} {importance_pct:.0f}% priority")
    
    # 4. Knowledge Entropy
    print(f"\n4️⃣  KNOWLEDGE DISTRIBUTION")
    entropy = state.knowledge_entropy
    print(f"   Knowledge Entropy: {entropy:.3f}")
    if entropy > 2.0:
        print(f"   → Knowledge is SPREAD OUT across many domains")
        print(f"   → Recommendation: CONSOLIDATE focus")
    else:
        print(f"   → Knowledge is CONCENTRATED in few domains")
        print(f"   → Recommendation: EXPLORE new domains")
    
    # 5. Time to Mastery
    print(f"\n5️⃣  LEARNING TIMELINE")
    ttm = state.estimated_time_to_mastery
    print(f"   Estimated Time to Mastery: {ttm:.0f} steps")
    
    # 6. Confidence Trajectory
    if len(state.performance_trajectory) > 0:
        print(f"\n6️⃣  PERFORMANCE TRAJECTORY (last 10 evaluations)")
        recent = state.performance_trajectory[-10:]
        for i, perf in enumerate(recent):
            bar = '█' * int(perf * 20) + '░' * (20 - int(perf * 20))
            print(f"   Step {i:2d}: {bar} {perf:.1%}")
    
    # 7. Bottleneck Analysis
    print(f"\n7️⃣  BOTTLENECK ANALYSIS")
    print(f"   What's limiting progress?")
    for bottleneck in state.current_bottlenecks:
        print(f"   • {bottleneck}")
    
    print("\n" + "="*70)


# ============================================================================
# VISUALIZATION: Create learning curves
# ============================================================================

def create_learning_visualization(aware_model):
    """
    Create visualization of learning progress across domains.
    """
    
    state = aware_model.get_awareness_state()
    
    print("\n" + "="*70)
    print("LEARNING PROGRESS VISUALIZATION")
    print("="*70)
    
    print(f"\nDomain Mastery Levels:")
    domains = ['vision', 'language', 'audio']
    for domain in domains:
        competence = state.confidence_by_domain.get(domain, 0.0)
        
        # 25-char bar
        filled = int(competence * 25)
        bar = '█' * filled + '░' * (25 - filled)
        
        # Status indicator
        if competence < 0.3:
            status = "❌ NEEDS LEARNING"
        elif competence < 0.6:
            status = "🔄 IN PROGRESS"
        elif competence < 0.8:
            status = "✓ DEVELOPING"
        else:
            status = "✅ MASTERED"
        
        print(f"  {domain:10s} {bar} {competence:5.1%}  {status}")
    
    # Overall progress
    avg_competence = np.mean(list(state.confidence_by_domain.values()))
    filled = int(avg_competence * 25)
    bar = '█' * filled + '░' * (25 - filled)
    print(f"\n  {'OVERALL':10s} {bar} {avg_competence:5.1%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Train the model
    aware_model, base_model = train_with_self_awareness(
        num_steps=2000,
        report_interval=250
    )
    
    # Analyze self-awareness
    analyze_self_awareness(aware_model)
    
    # Create visualization
    create_learning_visualization(aware_model)
    
    # Print final metrics
    state = aware_model.get_awareness_state()
    metrics = aware_model.self_awareness.get_awareness_metrics()
    
    print(f"\n" + "="*70)
    print("FINAL METRICS")
    print("="*70)
    print(f"Confidence:     {metrics['confidence']:.3f}")
    print(f"Competence:     {metrics['competence']:.3f}")
    print(f"Uncertainty:    {metrics['uncertainty']:.4f}")
    print(f"Knowledge Entropy: {metrics['knowledge_entropy']:.3f}")
    print(f"Learning Phase: {metrics['learning_phase']}")
    print("="*70)
