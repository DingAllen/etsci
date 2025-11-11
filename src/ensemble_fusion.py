"""
DS Ensemble: Dempster-Shafer Evidence Theory based Ensemble System
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os

from ds_theory import (
    softmax_to_mass, multi_source_fusion, pignistic_transform,
    compute_belief, compute_plausibility, get_uncertainty_interval
)


class DSEnsemble:
    """
    Dempster-Shafer Theory based Ensemble Classifier
    """
    
    def __init__(self, models, model_names, device='cpu', fusion_method='dempster',
                 belief_strategy='direct', temperature=1.0):
        """
        Initialize DS Ensemble
        
        Args:
            models: List of PyTorch models
            model_names: List of model names
            device: Device to run on
            fusion_method: Method for fusing mass functions
            belief_strategy: Strategy for belief assignment ('direct', 'temperature', 'sqrt')
            temperature: Temperature for scaling (if using temperature strategy)
        """
        self.models = models
        self.model_names = model_names
        self.device = device
        self.fusion_method = fusion_method
        self.belief_strategy = belief_strategy
        self.temperature = temperature
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
        Predict with uncertainty quantification using DS theory
        
        Args:
            inputs: Input tensor (batch_size, channels, height, width)
        
        Returns:
            predictions: Predicted classes (batch_size,)
            uncertainties: Dictionary with uncertainty metrics
            all_masses: List of mass functions for each sample
        """
        inputs = inputs.to(self.device)
        batch_size = inputs.size(0)
        
        # Collect predictions from all models
        all_probs = []
        for model in self.models:
            probs = self.predict_single_model(model, inputs)
            all_probs.append(probs.cpu().numpy())
        
        # Process each sample in the batch
        predictions = []
        all_masses = []
        uncertainties = {
            'belief': [],
            'plausibility': [],
            'interval_width': [],
            'conflict': [],
            'doubt': []
        }
        
        for i in range(batch_size):
            # Get mass functions from each model for this sample
            mass_functions = []
            for model_probs in all_probs:
                mass = softmax_to_mass(
                    model_probs[i], 
                    strategy=self.belief_strategy,
                    temperature=self.temperature
                )
                mass_functions.append(mass)
            
            # Fuse mass functions using DS theory
            fused_mass, conflicts = multi_source_fusion(
                mass_functions, 
                method=self.fusion_method
            )
            
            # Make decision using pignistic transformation
            final_probs = pignistic_transform(fused_mass, num_classes=self.num_classes)
            predicted_class = np.argmax(final_probs)
            
            # Compute uncertainty metrics for predicted class
            belief, plausibility, interval_width = get_uncertainty_interval(
                fused_mass, predicted_class
            )
            doubt = 1.0 - plausibility
            avg_conflict = np.mean(conflicts) if len(conflicts) > 0 else 0.0
            
            predictions.append(predicted_class)
            all_masses.append(fused_mass)
            
            uncertainties['belief'].append(belief)
            uncertainties['plausibility'].append(plausibility)
            uncertainties['interval_width'].append(interval_width)
            uncertainties['conflict'].append(avg_conflict)
            uncertainties['doubt'].append(doubt)
        
        predictions = np.array(predictions)
        for key in uncertainties:
            uncertainties[key] = np.array(uncertainties[key])
        
        return predictions, uncertainties, all_masses
    
    def predict(self, inputs):
        """Simple prediction without detailed uncertainty"""
        predictions, _, _ = self.predict_with_uncertainty(inputs)
        return predictions
    
    def evaluate(self, data_loader, return_details=False):
        """
        Evaluate ensemble on a dataset
        
        Args:
            data_loader: PyTorch DataLoader
            return_details: If True, return detailed results
        
        Returns:
            accuracy: Overall accuracy
            details: (if return_details=True) Dictionary with detailed metrics
        """
        all_predictions = []
        all_labels = []
        all_uncertainties = {
            'belief': [],
            'plausibility': [],
            'interval_width': [],
            'conflict': [],
            'doubt': []
        }
        
        for inputs, labels in tqdm(data_loader, desc='Evaluating DS Ensemble'):
            predictions, uncertainties, _ = self.predict_with_uncertainty(inputs)
            
            all_predictions.extend(predictions)
            all_labels.extend(labels.cpu().numpy())
            
            for key in all_uncertainties:
                all_uncertainties[key].extend(uncertainties[key])
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute accuracy
        accuracy = 100.0 * np.mean(all_predictions == all_labels)
        
        if return_details:
            # Convert to numpy arrays
            for key in all_uncertainties:
                all_uncertainties[key] = np.array(all_uncertainties[key])
            
            # Compute additional metrics
            correct_mask = all_predictions == all_labels
            
            details = {
                'accuracy': accuracy,
                'predictions': all_predictions,
                'labels': all_labels,
                'correct_mask': correct_mask,
                'uncertainties': all_uncertainties,
                'avg_belief': np.mean(all_uncertainties['belief']),
                'avg_plausibility': np.mean(all_uncertainties['plausibility']),
                'avg_interval_width': np.mean(all_uncertainties['interval_width']),
                'avg_conflict': np.mean(all_uncertainties['conflict']),
                'avg_doubt': np.mean(all_uncertainties['doubt']),
                # Uncertainty correlation with errors
                'conflict_on_errors': np.mean(all_uncertainties['conflict'][~correct_mask]) if (~correct_mask).sum() > 0 else 0,
                'conflict_on_correct': np.mean(all_uncertainties['conflict'][correct_mask]) if correct_mask.sum() > 0 else 0,
            }
            
            return accuracy, details
        
        return accuracy


class SimpleEnsemble:
    """Simple averaging/voting ensemble for baseline comparison"""
    
    def __init__(self, models, device='cpu', method='average'):
        """
        Args:
            models: List of models
            device: Device
            method: 'average' or 'vote'
        """
        self.models = models
        self.device = device
        self.method = method
        
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict(self, inputs):
        """Predict using ensemble"""
        inputs = inputs.to(self.device)
        
        all_preds = []
        with torch.no_grad():
            for model in self.models:
                outputs = model(inputs)
                if self.method == 'average':
                    probs = torch.softmax(outputs, dim=1)
                    all_preds.append(probs)
                else:  # voting
                    preds = outputs.argmax(dim=1)
                    all_preds.append(preds)
        
        if self.method == 'average':
            # Average probabilities
            avg_probs = torch.stack(all_preds).mean(dim=0)
            predictions = avg_probs.argmax(dim=1)
        else:
            # Majority voting
            votes = torch.stack(all_preds)
            predictions = torch.mode(votes, dim=0)[0]
        
        return predictions.cpu().numpy()
    
    def evaluate(self, data_loader):
        """Evaluate on dataset"""
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(data_loader, desc='Evaluating Simple Ensemble'):
            predictions = self.predict(inputs)
            labels = labels.cpu().numpy()
            
            correct += np.sum(predictions == labels)
            total += len(labels)
        
        accuracy = 100.0 * correct / total
        return accuracy


if __name__ == '__main__':
    print("DS Ensemble module loaded successfully")
    print("Use this module to create ensemble classifiers with DS theory")
