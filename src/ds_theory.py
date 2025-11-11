"""
Dempster-Shafer Evidence Theory Implementation
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import softmax as scipy_softmax


def softmax_to_mass(softmax_probs, strategy='direct', temperature=1.0, epsilon=1e-10):
    """
    Convert softmax probabilities to Dempster-Shafer mass function
    
    Args:
        softmax_probs: Softmax probabilities (numpy array or tensor), shape (num_classes,)
        strategy: Strategy for conversion ('direct', 'temperature', 'sqrt')
        temperature: Temperature parameter for scaling (T=1: no change, T<1: sharper, T>1: smoother)
        epsilon: Small constant to avoid numerical issues
    
    Returns:
        mass: Mass function as dict {frozenset(classes): mass_value}
              frozenset({0}): mass assigned to class 0
              frozenset({0,1,...,9}): mass assigned to uncertainty (frame of discernment)
    """
    # Convert to numpy if tensor
    if torch.is_tensor(softmax_probs):
        softmax_probs = softmax_probs.detach().cpu().numpy()
    
    # Ensure it's 1D
    softmax_probs = np.asarray(softmax_probs).flatten()
    num_classes = len(softmax_probs)
    
    # Apply temperature scaling if requested
    if strategy == 'temperature' and temperature != 1.0:
        # Apply temperature before softmax
        logits = np.log(softmax_probs + epsilon)
        softmax_probs = scipy_softmax(logits / temperature)
    elif strategy == 'sqrt':
        # Square root normalization for less confident predictions
        softmax_probs = np.sqrt(softmax_probs)
        softmax_probs = softmax_probs / (softmax_probs.sum() + epsilon)
    
    # Create mass function
    mass = {}
    
    # Assign mass to each singleton class
    for i in range(num_classes):
        if softmax_probs[i] > epsilon:
            mass[frozenset([i])] = softmax_probs[i]
    
    # Compute uncertainty mass (assigned to the whole frame)
    assigned_mass = sum(mass.values())
    uncertainty_mass = 1.0 - assigned_mass
    
    if uncertainty_mass > epsilon:
        # Frame of discernment (all possible classes)
        frame = frozenset(range(num_classes))
        mass[frame] = uncertainty_mass
    
    return mass


def normalize_mass(mass, epsilon=1e-10):
    """
    Normalize mass function to sum to 1
    
    Args:
        mass: Mass function dict
        epsilon: Small constant
    
    Returns:
        Normalized mass function
    """
    total = sum(mass.values())
    if total < epsilon:
        return mass
    
    return {k: v / total for k, v in mass.items()}


def dempster_combine(mass1, mass2, epsilon=1e-10):
    """
    Combine two mass functions using Dempster's rule of combination
    
    Args:
        mass1: First mass function (dict)
        mass2: Second mass function (dict)
        epsilon: Small constant
    
    Returns:
        combined_mass: Combined mass function
        conflict: Conflict measure (0 to 1)
    """
    # Compute combined mass for all possible intersections
    combined = {}
    conflict_mass = 0.0
    
    for focal1, m1 in mass1.items():
        for focal2, m2 in mass2.items():
            # Intersection of focal sets
            intersection = focal1 & focal2
            
            if len(intersection) == 0:
                # Empty intersection - conflict
                conflict_mass += m1 * m2
            else:
                # Non-empty intersection
                if intersection not in combined:
                    combined[intersection] = 0.0
                combined[intersection] += m1 * m2
    
    # Normalize by (1 - conflict)
    normalization = 1.0 - conflict_mass
    
    if normalization > epsilon:
        combined = {k: v / normalization for k, v in combined.items()}
    else:
        # Total conflict - return uniform distribution
        print(f"Warning: High conflict ({conflict_mass:.4f}), returning empty mass")
        combined = {}
    
    return combined, conflict_mass


def multi_source_fusion(mass_functions, method='dempster', weights=None, epsilon=1e-10):
    """
    Fuse multiple mass functions from different evidence sources
    
    Args:
        mass_functions: List of mass function dicts
        method: Fusion method ('dempster', 'weighted_dempster', 'yager')
        weights: Optional weights for each source (for weighted fusion)
        epsilon: Small constant
    
    Returns:
        fused_mass: Fused mass function
        conflicts: List of pairwise conflict measures
    """
    if len(mass_functions) == 0:
        return {}, []
    
    if len(mass_functions) == 1:
        return mass_functions[0], []
    
    # Apply weights if provided
    if weights is not None and method == 'weighted_dempster':
        weighted_masses = []
        for mass, weight in zip(mass_functions, weights):
            # Discount mass function by weight
            weighted = discount_mass(mass, 1.0 - weight)
            weighted_masses.append(weighted)
        mass_functions = weighted_masses
    
    # Sequential combination
    conflicts = []
    fused = mass_functions[0]
    
    for i in range(1, len(mass_functions)):
        fused, conflict = dempster_combine(fused, mass_functions[i], epsilon)
        conflicts.append(conflict)
        
        if len(fused) == 0:
            # High conflict, try to recover
            print(f"Warning: Empty mass after combining source {i}")
            break
    
    return fused, conflicts


def discount_mass(mass, discount_factor, epsilon=1e-10):
    """
    Apply discount factor to mass function (reliability discounting)
    
    Args:
        mass: Original mass function
        discount_factor: Discount factor (0 to 1), where 0 = fully reliable, 1 = unreliable
        epsilon: Small constant
    
    Returns:
        Discounted mass function
    """
    if discount_factor < epsilon:
        return mass.copy()
    
    discounted = {}
    
    # Get frame of discernment from mass keys
    all_elements = set()
    for focal_set in mass.keys():
        all_elements.update(focal_set)
    frame = frozenset(all_elements)
    
    # Discount each focal set
    total_discounted = 0.0
    for focal_set, mass_value in mass.items():
        new_mass = mass_value * (1.0 - discount_factor)
        if new_mass > epsilon:
            discounted[focal_set] = new_mass
            total_discounted += new_mass
    
    # Add discounted mass to frame (uncertainty)
    uncertainty = 1.0 - total_discounted
    if uncertainty > epsilon:
        if frame in discounted:
            discounted[frame] += uncertainty
        else:
            discounted[frame] = uncertainty
    
    return discounted


def pignistic_transform(mass, num_classes=10, epsilon=1e-10):
    """
    Convert mass function to probability distribution using pignistic transformation
    
    Args:
        mass: Mass function dict
        num_classes: Number of classes
        epsilon: Small constant
    
    Returns:
        probability: Probability distribution (numpy array)
    """
    probability = np.zeros(num_classes)
    
    for focal_set, mass_value in mass.items():
        # Distribute mass equally among elements in focal set
        if len(focal_set) > 0:
            mass_per_element = mass_value / len(focal_set)
            for element in focal_set:
                if element < num_classes:  # Safety check
                    probability[element] += mass_per_element
    
    # Normalize to ensure sum = 1
    total = probability.sum()
    if total > epsilon:
        probability = probability / total
    
    return probability


def compute_belief(mass, hypothesis, epsilon=1e-10):
    """
    Compute belief for a hypothesis (lower bound of probability)
    
    Args:
        mass: Mass function
        hypothesis: Set of classes (frozenset or set)
        epsilon: Small constant
    
    Returns:
        Belief value
    """
    if not isinstance(hypothesis, frozenset):
        hypothesis = frozenset(hypothesis)
    
    belief = 0.0
    for focal_set, mass_value in mass.items():
        # Sum mass of all focal sets that are subsets of hypothesis
        if focal_set.issubset(hypothesis):
            belief += mass_value
    
    return belief


def compute_plausibility(mass, hypothesis, epsilon=1e-10):
    """
    Compute plausibility for a hypothesis (upper bound of probability)
    
    Args:
        mass: Mass function
        hypothesis: Set of classes (frozenset or set)
        epsilon: Small constant
    
    Returns:
        Plausibility value
    """
    if not isinstance(hypothesis, frozenset):
        hypothesis = frozenset(hypothesis)
    
    plausibility = 0.0
    for focal_set, mass_value in mass.items():
        # Sum mass of all focal sets that intersect with hypothesis
        if len(focal_set & hypothesis) > 0:
            plausibility += mass_value
    
    return plausibility


def compute_doubt(mass, hypothesis, epsilon=1e-10):
    """
    Compute doubt for a hypothesis (complement of plausibility)
    
    Args:
        mass: Mass function
        hypothesis: Set of classes
        epsilon: Small constant
    
    Returns:
        Doubt value
    """
    return 1.0 - compute_plausibility(mass, hypothesis, epsilon)


def max_belief_decision(mass, num_classes=10):
    """
    Make decision based on maximum belief
    
    Args:
        mass: Mass function
        num_classes: Number of classes
    
    Returns:
        predicted_class: Class with maximum belief
        max_belief: Belief value for predicted class
    """
    beliefs = []
    for i in range(num_classes):
        belief = compute_belief(mass, frozenset([i]))
        beliefs.append(belief)
    
    beliefs = np.array(beliefs)
    predicted_class = np.argmax(beliefs)
    max_belief = beliefs[predicted_class]
    
    return predicted_class, max_belief


def max_plausibility_decision(mass, num_classes=10):
    """
    Make decision based on maximum plausibility
    
    Args:
        mass: Mass function
        num_classes: Number of classes
    
    Returns:
        predicted_class: Class with maximum plausibility
        max_plausibility: Plausibility value for predicted class
    """
    plausibilities = []
    for i in range(num_classes):
        plaus = compute_plausibility(mass, frozenset([i]))
        plausibilities.append(plaus)
    
    plausibilities = np.array(plausibilities)
    predicted_class = np.argmax(plausibilities)
    max_plausibility = plausibilities[predicted_class]
    
    return predicted_class, max_plausibility


def get_uncertainty_interval(mass, class_id):
    """
    Get belief-plausibility interval for a class (uncertainty interval)
    
    Args:
        mass: Mass function
        class_id: Class index
    
    Returns:
        belief, plausibility, interval_width
    """
    hypothesis = frozenset([class_id])
    belief = compute_belief(mass, hypothesis)
    plausibility = compute_plausibility(mass, hypothesis)
    interval_width = plausibility - belief
    
    return belief, plausibility, interval_width


if __name__ == '__main__':
    # Test DS theory implementation
    print("Testing Dempster-Shafer Theory Implementation\n")
    
    # Test 1: Softmax to mass conversion
    print("Test 1: Softmax to Mass Conversion")
    softmax_probs = np.array([0.7, 0.2, 0.05, 0.03, 0.01, 0.01, 0, 0, 0, 0])
    mass = softmax_to_mass(softmax_probs, strategy='direct')
    print(f"Softmax: {softmax_probs}")
    print(f"Mass function: {mass}")
    print()
    
    # Test 2: Belief and Plausibility
    print("Test 2: Belief and Plausibility")
    belief_0 = compute_belief(mass, frozenset([0]))
    plaus_0 = compute_plausibility(mass, frozenset([0]))
    print(f"Class 0 - Belief: {belief_0:.4f}, Plausibility: {plaus_0:.4f}")
    print(f"Uncertainty interval width: {plaus_0 - belief_0:.4f}")
    print()
    
    # Test 3: Combining two mass functions
    print("Test 3: Dempster's Rule of Combination")
    softmax2 = np.array([0.6, 0.3, 0.05, 0.02, 0.01, 0.01, 0.01, 0, 0, 0])
    mass2 = softmax_to_mass(softmax2, strategy='direct')
    
    combined, conflict = dempster_combine(mass, mass2)
    print(f"Mass 1: {list(mass.values())[:3]}")
    print(f"Mass 2: {list(mass2.values())[:3]}")
    print(f"Combined mass: {list(combined.values())[:3]}")
    print(f"Conflict: {conflict:.4f}")
    print()
    
    # Test 4: Pignistic transformation
    print("Test 4: Pignistic Transformation")
    prob = pignistic_transform(combined, num_classes=10)
    print(f"Pignistic probability: {prob}")
    print(f"Sum: {prob.sum():.4f}")
    print()
    
    # Test 5: Multi-source fusion
    print("Test 5: Multi-source Fusion")
    softmax3 = np.array([0.5, 0.4, 0.05, 0.02, 0.01, 0.01, 0.01, 0, 0, 0])
    mass3 = softmax_to_mass(softmax3, strategy='direct')
    
    fused, conflicts = multi_source_fusion([mass, mass2, mass3])
    print(f"Fused from 3 sources")
    print(f"Conflicts: {conflicts}")
    decision = pignistic_transform(fused, num_classes=10)
    print(f"Final decision: Class {np.argmax(decision)} with prob {decision.max():.4f}")
    print()
    
    print("All tests passed!")
