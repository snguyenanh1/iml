import logging
import os
import re
import json
from typing import Any, Dict, Optional
import optuna
from .base_agent import BaseAgent
from .utils import init_llm

logger = logging.getLogger(__name__)

class HyperparameterTuningAgent(BaseAgent):
    """
    Agent to perform hyperparameter tuning using Optuna.
    """
    def __init__(self, config: Any, manager: Any, llm_config: Any=None, **kwargs):
        super().__init__(config, manager)
        # Store config for iteration-specific access
        self.config = config
        # Initialize LLM and prompt handler if llm_config provided
        if llm_config:
            self.llm = init_llm(
                llm_config, agent_name='hyperparameter_tuning', multi_turn=False
            )
            from ..prompts.hyperparameter_tuning_prompt import HyperparameterTuningPrompt
            self.prompt_handler = HyperparameterTuningPrompt(
                manager=manager, llm_config=llm_config
            )

    def _extract_performance_metrics(self, stdout_content: str) -> Dict[str, float]:
        """Extract performance metrics from hyperparameter tuning stdout."""
        metrics = {}
        
        # Common metric patterns
        patterns = {
            'best_score': r'Best score:\s*([0-9]*\.?[0-9]+)',
            'best_params_score': r'Best parameters.*score[:\s]*([0-9]*\.?[0-9]+)',
            'final_score': r'Final.*score[:\s]*([0-9]*\.?[0-9]+)',
            'tuned_score': r'Tuned.*score[:\s]*([0-9]*\.?[0-9]+)',
            'validation_score': r'Validation Score:\s*([0-9]*\.?[0-9]+)',
            'cv_score': r'CV Score:\s*([0-9]*\.?[0-9]+)',
            'accuracy': r'Accuracy:\s*([0-9]*\.?[0-9]+)',
            'f1_score': r'F1[- ]?Score:\s*([0-9]*\.?[0-9]+)',
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, stdout_content, re.IGNORECASE)
            if matches:
                try:
                    # Take the last occurrence (most recent score)
                    metrics[metric_name] = float(matches[-1])
                except ValueError:
                    continue
        
        return metrics

    def _get_best_tuning_score(self, metrics: Dict[str, float]) -> Optional[float]:
        """Get the best performance score from tuning metrics."""
        # Priority order for score selection
        score_priority = ['best_score', 'best_params_score', 'final_score', 'tuned_score', 
                         'validation_score', 'cv_score', 'accuracy', 'f1_score']
        
        for score_name in score_priority:
            if score_name in metrics:
                return metrics[score_name]
        
        return None
    
    def _save_tuned_scores(self, tuning_metrics: Dict[str, float]):
        """Save tuned scores to separate tuned_scores.json file."""
        iteration_name = os.path.basename(self.manager.output_folder)
        tuned_file = os.path.join(self.manager.output_folder, "tuned_scores.json")
        
        try:
            # Create tuned scores data
            score_data = {
                "iteration_name": iteration_name,
                "iteration_type": self.manager.output_folder.split('_')[-1] if '_' in self.manager.output_folder else "unknown",
                "timestamp": "hyperparameter_tuned",
                "scores": tuning_metrics,
                "best_score": self._get_best_tuning_score(tuning_metrics),
                "submission_file": "submission_tuned.csv"
            }
            
            # Save tuned scores to separate file
            with open(tuned_file, 'w', encoding='utf-8') as f:
                json.dump(score_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Tuned scores saved to: {tuned_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save tuned scores to JSON: {e}")

    def _get_iteration_config(self, iteration_type: Optional[str] = None) -> Dict[str, Any]:
        """Get hyperparameter tuning configuration for the specific iteration type."""
        # Get hyperparameter tuning config
        hpt_config = getattr(self.config, 'hyperparameter_tuning', {})
        
        # If no iteration type specified or no iteration-specific config exists, use default
        if iteration_type is None or iteration_type not in hpt_config:
            default_config = hpt_config.get('default', {})
            logger.info(f"Using default hyperparameter tuning configuration")
            return default_config
        
        # Use iteration-specific configuration
        iteration_config = hpt_config.get(iteration_type, {})
        logger.info(f"Using {iteration_type}-specific hyperparameter tuning configuration")
        logger.info(f"Configuration: {iteration_config}")
        return iteration_config

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Executes hyperparameter tuning and returns best parameters.
        """
        self.manager.log_agent_start("Starting hyperparameter tuning phase...")

        # Get iteration-specific configuration
        iteration_config = self._get_iteration_config(iteration_type)
        
        # Log which configuration is being used
        if iteration_type:
            logger.info(f"🎯 Running hyperparameter tuning for iteration type: {iteration_type}")
        else:
            logger.info("🎯 Running hyperparameter tuning with default configuration")

        # The manager.output_folder is already set to the iteration folder during multi-iteration
        # e.g., /output/dog-breed-identification/iteration_2_custom_nn
        final_code_path = os.path.join(self.manager.output_folder, 'states', 'final_executable_code.py')
        
        if not os.path.exists(final_code_path):
            logger.error(f"final_executable_code.py not found at {final_code_path}. Cannot proceed with hyperparameter tuning.")
            return {"status": "failed", "error": f"final_executable_code.py not available at {final_code_path}"}
        
        # Read the successful code to pass to LLM for analysis
        try:
            with open(final_code_path, 'r', encoding='utf-8') as f:
                final_executable_code = f.read()
            logger.info(f"Successfully read final_executable_code.py from {final_code_path}")
        except Exception as e:
            logger.error(f"Failed to read final_executable_code.py: {e}")
            return {"status": "failed", "error": f"Failed to read final code: {e}"}

        # Get iteration-specific parameters
        max_retries = iteration_config.get('max_retries', 5)
        timeout = iteration_config.get('timeout', 3600)
        n_trials = iteration_config.get('n_trials', 25)
        
        # Log the configuration being used
        logger.info(f"📊 Hyperparameter tuning configuration:")
        logger.info(f"   - Trials: {n_trials}")
        logger.info(f"   - Timeout: {timeout}s ({timeout/3600:.1f}h)")
        logger.info(f"   - Max retries: {max_retries}")
        
        # Check for fast training mode (for deep learning)
        fast_training_mode = iteration_config.get('fast_training_mode', False)
        if fast_training_mode:
            logger.info(f"⚡ Fast training mode enabled for {iteration_type}")
            logger.info(f"   - Strategy: Reduced epochs/data for screening")
            logger.info(f"   - This allows more trials within time constraints")
        # Initial prompt-based script
        if not hasattr(self, 'prompt_handler'):
            logger.error("Prompt handler not configured for hyperparameter tuning.")
            return {"status": "failed", "error": "No prompt handler available for tuning."}
        
        # Pass the final executable code to LLM for analysis and tuning script generation
        # Use iteration-specific config for prompt building
        prompt = self.prompt_handler.build(iteration_config, final_executable_code)
        tuning_script = self.prompt_handler.parse(self.llm.assistant_chat(prompt))
        # Prepend working dir and path injection (no import needed as LLM will extract paths from code)
        injection = (
            "import os, sys\n"
            f"os.chdir(r'{self.manager.output_folder}')\n"
            f"sys.path.insert(0, r'{self.manager.output_folder}')\n"
        )
        tuning_script = injection + tuning_script
        # Execute tuning script with retry and repair via LLM on errors using iteration-specific max_retries
        last_stdout = None
        last_stderr = None
        for attempt in range(1, max_retries + 1):
            # Write and run
            result = self.manager.execute_code(
                code_to_execute=tuning_script,
                phase_name='hyperparameter_tuning',
                attempt=attempt
            )
            if result.get('success'):
                last_stdout = result.get('stdout')
                break
            # On failure, log and save for debugging
            last_stderr = result.get('stderr', '')
            error_to_log = last_stderr.split('\n')[-10:]
            # Prepare error text without backslashes in f-string expression
            error_text = "\n".join(error_to_log)
            self.manager.save_and_log_states(
                f"---ATTEMPT {attempt} FAILED---\nSCRIPT:\n{tuning_script}\n\nERROR:\n{error_text}",
                f"hpt_attempt_{attempt}_failed.log"
            )
            logger.warning(f"Hyperparameter tuning attempt {attempt} failed. Retrying...")
            # Build repair prompt for LLM with original code context
            repair_prompt = (
                f"The hyperparameter tuning script failed on attempt {attempt} with error:\n{last_stderr}\n"
                "Please fix the following Python script for hyperparameter tuning and return the corrected full script only.\n"
                "Original successful code for reference:\n```python\n" + final_executable_code + "\n```\n"
                "Failed tuning script:\n```python\n" + tuning_script + "\n```"
            )
            response = self.llm.assistant_chat(repair_prompt)
            tuning_script = self.prompt_handler.parse(response)
        else:
            logger.error(f"Hyperparameter tuning failed after {max_retries} attempts.")
            return {"status": "failed", "error": last_stderr}
        # Save tuning stdout from successful run
        self.manager.save_and_log_states(
            last_stdout or '', 'hyperparameter_tuning_stdout.txt'
        )
        
        # Extract and log performance metrics
        if last_stdout:
            tuning_metrics = self._extract_performance_metrics(last_stdout)
            tuning_score = self._get_best_tuning_score(tuning_metrics)
            
            # Save tuned scores to separate JSON file
            self._save_tuned_scores(tuning_metrics)
            
            # Get iteration type for clear logging
            iteration_name = os.path.basename(self.manager.output_folder)
            
            logger.info("🎯" + "="*60)
            logger.info(f"📊 HYPERPARAMETER TUNING COMPLETED: {iteration_name}")
            logger.info("🎯" + "="*60)
            
            if tuning_metrics:
                logger.info(f"📈 Extracted tuning metrics: {tuning_metrics}")
                if tuning_score is not None:
                    logger.info(f"🏆 Best tuned performance score: {tuning_score:.4f}")
                    
                    # Check if submission_tuned.csv was created
                    tuned_submission_path = os.path.join(self.manager.output_folder, "submission_tuned.csv")
                    if os.path.exists(tuned_submission_path):
                        logger.info(f"✅ Tuned submission created: submission_tuned.csv")
                        logger.info(f"📄 Tuned submission file: {iteration_name}/submission_tuned.csv")
                    else:
                        logger.warning(f"⚠️ Tuned submission file not found at {tuned_submission_path}")
                else:
                    logger.warning("⚠️ No performance score extracted from tuning output")
            else:
                logger.warning("⚠️ No performance metrics extracted from hyperparameter tuning")
            
            logger.info("🎯" + "="*60)
        
        self.manager.log_agent_end("Completed hyperparameter tuning phase.")
        return {"status": "success", "results": last_stdout, "metrics": tuning_metrics if 'tuning_metrics' in locals() else {}}
