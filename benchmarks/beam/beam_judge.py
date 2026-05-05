"""
BEAM Judge
=================
Evaluates the 10 ability types from the BEAM benchmark.
"""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# List of the 10 ability types BEAM tests
ABILITY_TYPES = [
    "temporal_tracking",
    "spatial_tracking",
    "coreference",
    "entity_tracking",
    "relation_tracking",
    "sentiment_tracking",
    "topic_tracking",
    "causality",
    "logical_reasoning",
    "state_tracking"
]

class BeamJudge:
    def __init__(self, eval_model: str):
        self.eval_model = eval_model
        # Ideally initialize LLM client for advanced judge matching here

    async def evaluate_answer(self, ability_type: str, question: str, expected: str, actual: str) -> float:
        """
        Evaluate a given answer according to its specific ability type.
        Can be extended with LLM-as-a-judge prompts.
        """
        # Naive exact/substring match for baseline
        if expected.strip().lower() in actual.strip().lower():
            return 1.0
        return 0.0

    def compute_summary(self, results: list[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = {at: {"correct": 0, "total": 0, "score": 0.0} for at in ABILITY_TYPES}
        
        for res in results:
            atype_raw = res.get("ability_type")
            atype = str(atype_raw) if atype_raw is not None else "unknown"
            
            if atype not in metrics:
                metrics[atype] = {"correct": 0, "total": 0, "score": 0.0}
            
            metrics[atype]["total"] += 1
            if res.get("score", 0.0) >= 0.5:
                metrics[atype]["correct"] += 1
                
        # Compute final accuracies
        for atype, stats in metrics.items():
            if stats["total"] > 0:
                stats["accuracy"] = stats["correct"] / stats["total"]
                
        return metrics
