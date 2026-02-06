import json
import os
from pathlib import Path
from typing import Any, List, Dict

class MemoryManager:
    def __init__(self, output_dir="outputs/memory"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, problem_type: str) -> Path:
        safe_name = problem_type.lower().strip().replace(" ", "_")
        return self.output_dir / f"{safe_name}.json"
    
    def load_memory(self, problem_type: str, problem_name: str = None) -> List[Dict[str, Any]]:
        file_path = self._get_file_path(problem_type)
        if not file_path.exists():
            return []
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            if problem_name:
                return [entry for entry in data if entry.get("problem_name") == problem_name]
            return data
        except Exception:
            return []
        
    def save_memory(self, problem_type: str, entry: Dict[str, Any]):
        file_path = self._get_file_path(problem_type)
        
        full_data = self.load_memory(problem_type)

        clean_entry = {
            "problem_name": entry.get("problem"),
            "method": entry.get("Method"),
            "parameters": entry.get("Parameters"),
            "score": entry.get("F1_Score") or entry.get("Silhouette") or entry.get("Fitness", 0.0),
            "metric_name": "F1" if "F1_Score" in entry else ("Silhouette" if "Silhouette" in entry else "Fitness"),
            "timestamp": entry.get("Timestamp")
        }

        full_data.append(clean_entry)

        with open(file_path, "w") as f:
            json.dump(full_data, f, indent=4)

    def get_context_string(self, problem_type: str, problem_name: str, top_k: int = 3) -> str:

        history = self.load_memory(problem_type, problem_name)

        if not history:
            return ""
        
        reverse_sort = True
        if problem_type == "combinatorial_optimization":
            reverse_sort = False

        history_sorted = sorted(history, key=lambda x: x["score"], reverse=reverse_sort)

        best_runs = history_sorted[:top_k]
        worst_runs = history_sorted[-1] if len(history_sorted) > top_k else None

        context = f"\n[MEMORY - BEST PAST CONFIGURATIONS FOR {problem_name}]:\n"
        for i, m in enumerate(best_runs, 1):
            context += (
                f"   - Rank {i}: {m['method']} | Params: {m['parameters']} "
                f"| {m['metric_name']}: {m['score']:.4f}\n"
            )
        
        if worst_runs:
            context += (
                f"\n[MEMORY - CONFIGURATION TO AVOID]:\n"
                f"   - Bad Run: {worst_runs['method']} | Params: {worst_runs['parameters']} "
                f"| Score: {worst_runs['score']:.4f}\n"
            )

        context += "   (Use these insights to refine your strategy.)\n"
        return context