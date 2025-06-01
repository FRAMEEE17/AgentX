#!/usr/bin/env python3
"""
Enhanced test runner for Lab 2 with detailed debugging and analysis.
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Tuple
from pathlib import Path
from dataclasses import dataclass

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.main import main_workflow, get_problem_and_code_from_taskpath, get_unit_tests_from_taskpath, get_task_lean_template_from_taskpath
from src.lean_runner import execute_lean_code


@dataclass
class TaskResult:
    """Detailed result for a single task."""
    task_id: str
    success: bool = False
    implementation_works: bool = False
    proof_works: bool = False
    runtime: float = 0.0
    generated_code: str = ""
    generated_proof: str = ""
    implementation_error: str = ""
    proof_error: str = ""
    approach_used: str = ""


class EnhancedTestRunner:
    """Enhanced test runner with debugging capabilities."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: Dict[str, TaskResult] = {}
        
    def run_single_task(self, task_id: str) -> TaskResult:
        """Run a single task with detailed analysis."""
        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Testing Task {task_id}")
            print(f"{'=' * 60}")
        
        result = TaskResult(task_id=task_id)
        task_folder = f"tasks/{task_id}"
        
        try:
            # Read task files
            if self.verbose:
                print(f"📁 Reading task files from {task_folder}...")
            
            problem_description, lean_code_template = get_problem_and_code_from_taskpath(task_folder)
            unit_tests = get_unit_tests_from_taskpath(task_folder)
            
            if self.verbose:
                print(f"📝 Problem description: {len(problem_description)} chars")
                print(f"🔧 Lean template: {len(lean_code_template)} chars")
                print(f"🧪 Unit tests: {len(unit_tests)} chars")
            
            # Generate solution
            if self.verbose:
                print(f"\n🤖 Generating solution...")
            
            start_time = time.time()
            solution = main_workflow(problem_description, lean_code_template)
            end_time = time.time()
            
            result.runtime = end_time - start_time
            result.generated_code = solution.get("code", "")
            result.generated_proof = solution.get("proof", "")
            
            if self.verbose:
                print(f"⏱️  Generation time: {result.runtime:.2f}s")
                print(f"💻 Generated code ({len(result.generated_code)} chars): {result.generated_code[:100]}...")
                print(f"🔍 Generated proof ({len(result.generated_proof)} chars): {result.generated_proof[:100]}...")
            
            # Test implementation only
            if self.verbose:
                print(f"\n🔬 Testing implementation (proof=sorry)...")
            
            result.implementation_works, result.implementation_error = self._test_implementation(
                lean_code_template, result.generated_code, unit_tests
            )
            
            if self.verbose:
                status = "✅ PASS" if result.implementation_works else "❌ FAIL"
                print(f"Implementation test: {status}")
                if result.implementation_error:
                    print(f"Implementation error: {result.implementation_error[:200]}...")
            
            # Test full solution
            if self.verbose:
                print(f"\n🔍 Testing full solution (implementation + proof)...")
            
            result.proof_works, result.proof_error = self._test_full_solution(
                lean_code_template, result.generated_code, result.generated_proof, unit_tests
            )
            
            if self.verbose:
                status = "✅ PASS" if result.proof_works else "❌ FAIL"
                print(f"Proof test: {status}")
                if result.proof_error:
                    print(f"Proof error: {result.proof_error[:200]}...")
            
            # Determine overall success
            result.success = result.implementation_works and result.proof_works
            
            if self.verbose:
                overall_status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"\n🎯 Overall result: {overall_status}")
        
        except Exception as e:
            if self.verbose:
                print(f"💥 Unexpected error: {e}")
            result.implementation_error = str(e)
        
        return result
    
    def _test_implementation(self, template: str, code: str, unit_tests: str) -> Tuple[bool, str]:
        """Test implementation with proof set to sorry."""
        try:
            test_code = template.replace("{{code}}", code).replace("{{proof}}", "sorry")
            full_test = test_code + f"\n\n{unit_tests}"
            
            # Skip Lean execution if in development mode
            if os.getenv("SKIP_LEAN_VERIFICATION", "").lower() == "true":
                return True, ""
            
            result = execute_lean_code(full_test)
            
            if "executed successfully" in result:
                return True, ""
            else:
                error = result.split("Lean Error:")[-1].strip() if "Lean Error:" in result else result
                return False, error
                
        except Exception as e:
            return False, str(e)
    
    def _test_full_solution(self, template: str, code: str, proof: str, unit_tests: str) -> Tuple[bool, str]:
        """Test full solution with implementation and proof."""
        try:
            test_code = template.replace("{{code}}", code).replace("{{proof}}", proof)
            full_test = test_code + f"\n\n{unit_tests}"
            
            # Skip Lean execution if in development mode
            if os.getenv("SKIP_LEAN_VERIFICATION", "").lower() == "true":
                return True, ""
            
            result = execute_lean_code(full_test)
            
            if "executed successfully" in result:
                return True, ""
            else:
                error = result.split("Lean Error:")[-1].strip() if "Lean Error:" in result else result
                return False, error
                
        except Exception as e:
            return False, str(e)
    
    def run_all_tasks(self) -> Dict[str, TaskResult]:
        """Run all public test tasks."""
        task_ids = ["task_id_0", "task_id_58", "task_id_77", "task_id_127", "task_id_227", 
                   "task_id_404", "task_id_431", "task_id_433", "task_id_435", "task_id_441", "task_id_447"]
        
        print(f"🚀 Starting test of {len(task_ids)} tasks")
        print(f"Tasks: {', '.join(task_ids)}")
        
        for task_id in task_ids:
            result = self.run_single_task(task_id)
            self.results[task_id] = result
        
        self._print_summary()
        return self.results
    
    def _print_summary(self):
        """Print test summary."""
        print(f"\n{'=' * 80}")
        print(f"TEST SUMMARY")
        print(f"{'=' * 80}")
        
        total_tasks = len(self.results)
        successful_implementations = sum(1 for r in self.results.values() if r.implementation_works)
        successful_proofs = sum(1 for r in self.results.values() if r.proof_works)
        fully_successful = sum(1 for r in self.results.values() if r.success)
        
        print(f"📊 Overall Statistics:")
        print(f"   Total tasks: {total_tasks}")
        print(f"   Working implementations: {successful_implementations}/{total_tasks} ({successful_implementations/total_tasks*100:.1f}%)")
        print(f"   Working proofs: {successful_proofs}/{total_tasks} ({successful_proofs/total_tasks*100:.1f}%)")
        print(f"   Fully successful: {fully_successful}/{total_tasks} ({fully_successful/total_tasks*100:.1f}%)")
        
        total_runtime = sum(r.runtime for r in self.results.values())
        avg_runtime = total_runtime / total_tasks if total_tasks > 0 else 0
        print(f"   Average runtime: {avg_runtime:.2f}s")
        print(f"   Total runtime: {total_runtime:.2f}s")
        
        print(f"\n📋 Task-by-Task Results:")
        for task_id, result in self.results.items():
            impl_status = "✅" if result.implementation_works else "❌"
            proof_status = "✅" if result.proof_works else "❌" 
            overall_status = "✅" if result.success else "❌"
            
            print(f"   {task_id:15} | Impl: {impl_status} | Proof: {proof_status} | Overall: {overall_status} | Time: {result.runtime:.2f}s")
        
        # Show failures
        failed_tasks = [task_id for task_id, result in self.results.items() if not result.success]
        if failed_tasks:
            print(f"\n❌ Failed Tasks ({len(failed_tasks)}):")
            for task_id in failed_tasks:
                result = self.results[task_id]
                print(f"   {task_id}:")
                if not result.implementation_works:
                    print(f"      Implementation: {result.implementation_error[:100]}...")
                if not result.proof_works:
                    print(f"      Proof: {result.proof_error[:100]}...")
        
        print(f"\n{'=' * 80}")


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Lab 2 Test Runner")
    parser.add_argument("--task", type=str, help="Test specific task (e.g., task_id_0)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    runner = EnhancedTestRunner(verbose=not args.quiet)
    
    if args.task:
        # Test single task
        result = runner.run_single_task(args.task)
        print(f"\nResult: {'SUCCESS' if result.success else 'FAILED'}")
        return result.success
    else:
        # Test all tasks
        results = runner.run_all_tasks()
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        print(f"\nFinal Score: {success_count}/{total_count} tasks passed")
        return success_count == total_count


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)