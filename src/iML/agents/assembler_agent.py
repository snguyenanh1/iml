# src/iML/agents/assembler_agent.py
import logging
import os
import json
import re
from typing import Dict, Any

from .base_agent import BaseAgent
from ..prompts import AssemblerPrompt
from .utils import init_llm

logger = logging.getLogger(__name__)

class AssemblerAgent(BaseAgent):
    """
    Agent to assemble, finalize, execute and fix final code.
    """
    def __init__(self, config: Dict, manager: Any, llm_config: Dict, max_retries: int = 10):
        super().__init__(config, manager)
        self.llm_config = llm_config
        self.llm = init_llm(
            llm_config=llm_config,
            agent_name="assembler",
            multi_turn=llm_config.get("multi_turn", False),
        )
        self.prompt_handler = AssemblerPrompt(
            manager=manager, 
            llm_config=self.llm_config
        )
        self.max_retries = max_retries

    def _extract_and_save_scores(self, stdout_content: str, iteration_type: str = None) -> Dict[str, float]:
        """Extract scores from stdout and save to non_tuned_scores.json file."""
        scores = {}
        
        # Simple patterns for common score formats
        patterns = {
            'validation_score': r'Validation Score:?\s*([0-9]*\.?[0-9]+)',
            'accuracy': r'Accuracy:?\s*([0-9]*\.?[0-9]+)',
            'f1_score': r'F1[- ]?Score:?\s*([0-9]*\.?[0-9]+)',
            'rmse': r'RMSE:?\s*([0-9]*\.?[0-9]+)',
            'mae': r'MAE:?\s*([0-9]*\.?[0-9]+)',
            'r2_score': r'R2[- ]?Score:?\s*([0-9]*\.?[0-9]+)',
            'auc': r'AUC:?\s*([0-9]*\.?[0-9]+)',
        }
        
        for metric_name, pattern in patterns.items():
            matches = re.findall(pattern, stdout_content, re.IGNORECASE)
            if matches:
                try:
                    scores[metric_name] = float(matches[-1])
                except ValueError:
                    continue
        
        # Save original (non-tuned) scores to separate JSON file
        iteration_name = os.path.basename(self.manager.output_folder)
        non_tuned_file = os.path.join(self.manager.output_folder, "non_tuned_scores.json")
        
        score_data = {
            "iteration_name": iteration_name,
            "iteration_type": iteration_type,
            "timestamp": "original_submission",
            "scores": scores,
            "submission_file": "submission.csv"
        }
        
        try:
            with open(non_tuned_file, 'w', encoding='utf-8') as f:
                json.dump(score_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Non-tuned scores saved to: {non_tuned_file}")
        except Exception as e:
            logger.warning(f"Failed to save non-tuned scores to JSON: {e}")
        
        return scores

    def __call__(self, iteration_type=None) -> Dict[str, Any]:
        """
        Assemble, execute and retry final code until successful.
        """
        self.manager.log_agent_start("Starting assembly and testing of final code...")

        preprocessing_code = self.manager.preprocessing_code
        modeling_code = self.manager.modeling_code
        description = self.manager.description_analysis

        if not preprocessing_code or not modeling_code:
            error = "Preprocessing or modeling code not available."
            logger.error(error)
            return {"status": "failed", "error": error}

        # Combine initial code
        combined_code = preprocessing_code + "\n\n" + modeling_code
        submission_path = os.path.join(self.manager.output_folder, "submission.csv")
        error_message = None

        for attempt in range(self.max_retries):
            logger.info(f"Assembly and execution attempt {attempt + 1}/{self.max_retries}...")

            # 1. Assemble/Fix code
            # On first attempt, error_message is None, LLM will just assemble.
            # In subsequent attempts, LLM will fix errors.
            prompt = self.prompt_handler.build(
                original_code=combined_code,
                output_path=submission_path,
                description=description,
                error_message=error_message,
                iteration_type=iteration_type
            )

            response = self.llm.assistant_chat(prompt)
            self.manager.save_and_log_states(
                content=response,
                save_name=f"assembler_raw_response_attempt_{attempt + 1}.txt",
            )

            final_code = self.prompt_handler.parse(response)
            
            # 2. Execute final code
            execution_result = self.manager.execute_code(final_code, "assembler", attempt + 1)
            
            if execution_result["success"]:
                logger.info("Final code executed successfully!")
                logger.info(f"Submission file created at: {submission_path}")
                self.manager.save_and_log_states(final_code, "final_executable_code.py")
                
                # Extract and save scores to JSON
                stdout_content = execution_result.get("stdout", "")
                scores = {}
                if stdout_content:
                    scores = self._extract_and_save_scores(stdout_content, iteration_type)
                
                # Log simple completion message (like original)
                iteration_name = os.path.basename(self.manager.output_folder)
                logger.info("🔥" + "="*60)
                logger.info(f"📊 ASSEMBLER EXECUTION COMPLETED: {iteration_name}")
                logger.info("🔥" + "="*60)
                
                if scores:
                    logger.info(f"📈 Extracted performance metrics: {scores}")
                    # Find primary score
                    primary_score = None
                    score_priority = ['validation_score', 'accuracy', 'f1_score', 'auc', 'r2_score']
                    for score_name in score_priority:
                        if score_name in scores:
                            primary_score = scores[score_name]
                            logger.info(f"🏆 Primary performance score: {primary_score:.4f}")
                            break
                    if primary_score is None:
                        logger.warning("⚠️ No primary score identified from metrics")
                else:
                    logger.warning("⚠️ No performance metrics extracted from execution output")
                
                logger.info(f"✅ Original submission created: submission.csv")
                logger.info(f"📄 Original submission file: {iteration_name}/submission.csv")
                logger.info("🔥" + "="*60)
                
                self.manager.log_agent_end("Completed assembly and execution of code.")
                return {"status": "success", "code": final_code, "submission_path": submission_path, "scores": scores}
            else:
                error_message = execution_result["stderr"]
                last_10_lines = error_message.split('\n')[-10:]
                error_to_log = '\n'.join(last_10_lines)
                logger.warning(f"Code execution failed on attempt {attempt + 1}. Error: {error_message}")
                self.manager.save_and_log_states(
                    f"---ATTEMPT {attempt+1}---\nCODE:\n{final_code}\n\nERROR:\n{error_to_log}",
                    f"assembler_attempt_{attempt+1}_failed.log"
                )
                # Update combined_code so next iteration LLM will fix the latest error version
                combined_code = final_code

        logger.error(f"Unable to generate working code after {self.max_retries} attempts.")
        self.manager.log_agent_end("Code assembly failed.")
        return {"status": "failed", "error": "Exceeded maximum retry attempts to generate code."}