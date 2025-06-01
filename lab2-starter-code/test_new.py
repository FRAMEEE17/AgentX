#!/usr/bin/env python3
"""
test script for Lab 2 with detailed analysis and debugging.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import main_workflow, get_problem_and_code_from_taskpath, get_unit_tests_from_taskpath, get_task_lean_template_from_taskpath
from src.lean_runner import execute_lean_code


class TestResult:
    """Class to store detailed test results."""
    
    def __init__(self, task_id: str):
        self.task_id = task_id
        self.passes_unit_tests = False
        self.proof_is_correct = False
        self.runtime = 0.0
        self.generated_code = ""
        self.generated_proof = ""
        self.implementation_error = ""
        self.proof_error = ""
        self.approach_used = ""
        self.success_stage = "none"  # none, implementation, proof, complete


def analyze_task_complexity(problem_description: str, task_lean_code: str) -> Dict[str, Any]:
    """Analyze the complexity of a task."""
    complexity = {
        "function_name": "",
        "input_types": [],
        "output_type": "",
        "has_arrays": False,
        "has_conditionals": False,
        "complexity_score": 1
    }
    
    # Extract function name
    import re
    func_match = re.search(r'def\s+(\w+)', task_lean_code)
    if func_match:
        complexity["function_name"] = func_match.group(1)
    
    # Check for arrays
    if "Array" in task_lean_code:
        complexity["has_arrays"] = True
        complexity["complexity_score"] += 2
    
    # Check for conditionals in description
    conditional_keywords = ["if", "condition", "check", "compare", "minimum", "maximum"]
    if any(keyword in problem_description.lower() for keyword in conditional_keywords):
        complexity["has_conditionals"] = True
        complexity["complexity_score"] += 1
    
    # Check for multiple parameters
    param_count = task_lean_code.count(":")
    if param_count > 3:  # More than simple single parameter
        complexity["complexity_score"] += 1
    
    return complexity


def test_single_task(task_id: str, verbose: bool = True) -> TestResult:
    """Test a single task with detailed analysis."""
    task_folder = f"tasks/{task_id}"
    result = TestResult(task_id)
    
    if verbose:
        print