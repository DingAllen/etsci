"""
Deep Ensemble Baseline Implementation
Following Lakshminarayanan et al., 2017
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm


class DeepEnsemble:
    """
    Deep Ensemble baseline: average softmax outputs from multiple models
    This is the gold standard for uncertainty quantification in deep learning
    
    Reference: Lakshminarayanan et al., "Simple and Scalable Predictive 
               Uncertainty Estimation using Deep Ensembles", NeurIPS 2017
    """
    
    def __init__(self, models, model_names, device='cpu'):
        """
        Initialize Deep Ensemble
        
        Args:
            models: List of PyTorch models (independently trained)
            model_names: List of model names
            device: Device to run on
        """
        self.models = models
        self.model_names = model_names
        self.device = device
        self.num_classes = 10
        
        # Move all models to device and set to eval mode
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict_single_model(self, model, inputs):
        """Get prediction from a single model"""
        with torch.no_grad():
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
        return probs
    
    def predict_with_uncertainty(self, inputs):
        """
        Predict with uncertainty quantification using Deep Ensemble
        
        Args:
            inputs: Input tensor (batch_size, channels, height, width)
        
        Returns:
            predictions: Predicted classes (batch_size,)
            uncertainties: Dictionary with uncertainty metrics
                - mean_probs: Mean predicted probabilities
                - predictive_entropy: Entropy of mean prediction
                - mutual_information: Mutual information (epistemic uncertainty)
                - variance: Variance in predictions across models
        """
        inputs = inputs.to(self.device)
        batch_size = inputs.size(0)
        
        # Collect predictions from all models
        all_probs = []
        for model in self.models:
            probs = self.predict_single_model(model, inputs)
            all_probs.append(probs.cpu().numpy())
        
        # Stack: (num_models, batch_size, num_classes)
        all_probs = np.stack(all_probs, axis=0)
        
        # Mean prediction (Deep Ensemble prediction)
        mean_probs = np.mean(all_probs, axis=0)  # (batch_size, num_classes)
        
        # Predicted classes
        predictions = np.argmax(mean_probs, axis=1)
        
        # Compute uncertainty metrics
        uncertainties = {}
        uncertainties['mean_probs'] = mean_probs
        
        # 1. Predictive Entropy (total uncertainty)
        # H[y|x] = - sum_c p(c) log p(c)
        epsilon = 1e-10
        predictive_entropy = -np.sum(
            mean_probs * np.log(mean_probs + epsilon), 
            axis=1
        )
        uncertainties['predictive_entropy'] = predictive_entropy
        
        # 2. Expected Entropy (aleatoric uncertainty)
        # E_theta[H[y|x,theta]] = E[- sum_c p(c|theta) log p(c|theta)]
        model_entropies = -np.sum(
            all_probs * np.log(all_probs + epsilon),
            axis=2
        )  # (num_models, batch_size)
        expected_entropy = np.mean(model_entropies, axis=0)
        uncertainties['expected_entropy'] = expected_entropy
        
        # 3. Mutual Information (epistemic uncertainty)
        # MI = H[y|x] - E_theta[H[y|x,theta]]
        mutual_information = predictive_entropy - expected_entropy
        uncertainties['mutual_information'] = mutual_information
        
        # 4. Variance (measure of disagreement)
        variance = np.var(all_probs, axis=0)  # (batch_size, num_classes)
        max_variance = np.max(variance, axis=1)  # max variance across classes
        uncertainties['variance'] = max_variance
        
        # 5. Confidence (max probability)
        confidence = np.max(mean_probs, axis=1)
        uncertainties['confidence'] = confidence
        
        return predictions, uncertainties
    
    def evaluate(self, data_loader, return_details=False):
        """
        Evaluate Deep Ensemble on a dataset
        
        Args:
            data_loader: PyTorch DataLoader
            return_details: Whether to return detailed metrics
        
        Returns:
            accuracy: Classification accuracy (%)
            details: Dictionary with detailed metrics (if return_details=True)
        """
        all_predictions = []
        all_labels = []
        all_uncertainties = {
            'predictive_entropy': [],
            'mutual_information': [],
            'variance': [],
            'confidence': [],
            'mean_probs': []
        }
        
        for inputs, labels in tqdm(data_loader, desc="Evaluating Deep Ensemble"):
            predictions, uncertainties = self.predict_with_uncertainty(inputs)
            
            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            
            for key in all_uncertainties:
                if key == 'mean_probs':
                    all_uncertainties[key].append(uncertainties[key])
                else:
                    all_uncertainties[key].extend(uncertainties[key].tolist())
        
        # Compute accuracy
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        accuracy = 100.0 * np.mean(all_predictions == all_labels)
        
        if not return_details:
            return accuracy
        
        # Prepare detailed metrics
        details = {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'mean_probs': np.vstack(all_uncertainties['mean_probs']),
            'predictive_entropy': np.array(all_uncertainties['predictive_entropy']),
            'mutual_information': np.array(all_uncertainties['mutual_information']),
            'variance': np.array(all_uncertainties['variance']),
            'confidence': np.array(all_uncertainties['confidence']),
        }
        
        # Compute average uncertainty
        details['avg_predictive_entropy'] = np.mean(details['predictive_entropy'])
        details['avg_mutual_information'] = np.mean(details['mutual_information'])
        details['avg_variance'] = np.mean(details['variance'])
        
        # Compute uncertainty on correct vs incorrect predictions
        correct_mask = (all_predictions == all_labels)
        details['entropy_on_correct'] = np.mean(details['predictive_entropy'][correct_mask])
        details['entropy_on_errors'] = np.mean(details['predictive_entropy'][~correct_mask])
        details['mi_on_correct'] = np.mean(details['mutual_information'][correct_mask])
        details['mi_on_errors'] = np.mean(details['mutual_information'][~correct_mask])
        
        return accuracy, details


def compare_deep_ensemble_vs_ds(deep_ensemble, ds_ensemble, data_loader):
    """
    Compare Deep Ensemble vs DS Ensemble
    
    Args:
        deep_ensemble: DeepEnsemble instance
        ds_ensemble: DSEnsemble instance
        data_loader: DataLoader for evaluation
    
    Returns:
        comparison: Dictionary with comparison metrics
    """
    print("Evaluating Deep Ensemble...")
    de_acc, de_details = deep_ensemble.evaluate(data_loader, return_details=True)
    
    print("Evaluating DS Ensemble...")
    ds_acc, ds_details = ds_ensemble.evaluate(data_loader, return_details=True)
    
    comparison = {
        'deep_ensemble': {
            'accuracy': de_acc,
            'avg_entropy': de_details['avg_predictive_entropy'],
            'avg_mi': de_details['avg_mutual_information'],
            'entropy_on_errors': de_details['entropy_on_errors'],
            'mi_on_errors': de_details['mi_on_errors'],
        },
        'ds_ensemble': {
            'accuracy': ds_acc,
            'avg_conflict': ds_details['avg_conflict'],
            'avg_interval_width': ds_details.get('avg_interval_width', 0),
            'conflict_on_errors': ds_details['conflict_on_errors'],
        }
    }
    
    return comparison, de_details, ds_details
