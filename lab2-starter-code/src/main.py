import os
import re
import json
import time
import functools
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.agents import Reasoning_Agent, LLM_Agent
from src.lean_runner import execute_lean_code
from src.embedding_db import VectorDB
from src.embedding_models import OpenAIEmbeddingModel


def timer_decorator(func_name: str):
    """Decorator to time function execution"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            print(f"⏱️  [{func_name}] took {duration:.3f}s")
            return result
        return wrapper
    return decorator


class PerformanceProfiler:
    """Tracks performance bottlenecks across the workflow"""
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing and record duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.timings[operation] = duration
            print(f"⏱️  [{operation}] took {duration:.3f}s")
            del self.start_times[operation]
            return duration
        return 0
    
    def get_summary(self) -> str:
        """Get performance summary"""
        if not self.timings:
            return "No timing data recorded"
        
        total_time = sum(self.timings.values())
        summary = f"\n📊 PERFORMANCE BREAKDOWN (Total: {total_time:.3f}s):\n"
        
        # Sort by duration (longest first)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)
        
        for operation, duration in sorted_timings:
            percentage = (duration / total_time) * 100
            summary += f"   {operation:<30} {duration:>6.3f}s ({percentage:>5.1f}%)\n"
        
        return summary


# Global profiler instance
profiler = PerformanceProfiler()


class ReasoningPath(Enum):
    DIRECT_IMPLEMENTATION = "direct_implementation"
    MATHLIB_APPROACH = "mathlib_approach" 
    SPECIFICATION_DRIVEN = "specification_driven"
    PATTERN_MATCHING = "pattern_matching"


@dataclass
class ThoughtNode:
    """A node in the Tree of Thoughts"""
    path: ReasoningPath
    reasoning: str
    code_approach: str
    proof_strategy: str
    confidence: float
    rationale: str


@dataclass
class Solution:
    """Generated solution with metadata"""
    code: str
    proof: str
    approach: str
    reasoning_trace: List[str]
    success: bool = False
    error: str = ""


class PlanningAgent:
    """Agent 1: Handles task decomposition and strategy planning"""
    
    def __init__(self):
        self.agent = Reasoning_Agent(model="o3-mini")
        self.previous_attempts = []
    
    @timer_decorator("PlanningAgent.analyze_problem")
    def analyze_problem(self, problem_description: str, task_lean_code: str) -> Dict[str, Any]:
        """Analyze problem and create strategic plan"""
        
        profiler.start_timer("Planning.extract_info")
        # Extract problem characteristics
        problem_info = self._extract_problem_info(problem_description, task_lean_code)
        profiler.end_timer("Planning.extract_info")
        
        profiler.start_timer("Planning.llm_call")
        # Create planning prompt
        planning_prompt = f"""
        You are a Lean 4 theorem proving strategist. Analyze this problem and create a solving strategy.

        PROBLEM DESCRIPTION:
        {problem_description}

        FUNCTION SIGNATURE: {problem_info['function_name']} {problem_info['parameters']} : {problem_info['return_type']}
        SPECIFICATION: {problem_info['specification']}
        PROBLEM TYPE: {problem_info['problem_type']}

        PREVIOUS ATTEMPTS: {len(self.previous_attempts)} failed attempts

        Create a strategic plan including:
        1. Implementation approach (direct, library-based, pattern-matching)
        2. Proof strategy (constructor, cases, simp, induction)
        3. Key insights about the problem structure
        4. Potential pitfalls to avoid

        Respond in JSON format with keys: approach, proof_strategy, insights, pitfalls
        """
        
        try:
            response = self.agent.get_response([{"role": "user", "content": planning_prompt}])
            profiler.end_timer("Planning.llm_call")
            
            profiler.start_timer("Planning.parse_response")
            # Parse JSON response
            plan = self._parse_json_response(response)
            plan.update(problem_info)
            profiler.end_timer("Planning.parse_response")
            return plan
        except Exception as e:
            profiler.end_timer("Planning.llm_call")
            print(f"[Planning Agent] Error: {e}")
            return problem_info
    
    def _extract_problem_info(self, problem_description: str, task_lean_code: str) -> Dict[str, str]:
        """Extract key problem information"""
        info = {
            "function_name": "",
            "parameters": "",
            "return_type": "", 
            "specification": "",
            "problem_type": "",
            "complexity": "medium"
        }
        
        # Extract function signature
        func_match = re.search(r'def\s+(\w+)\s*([^:]*)\s*:\s*([^:=]+)', task_lean_code)
        if func_match:
            info["function_name"] = func_match.group(1).strip()
            info["parameters"] = func_match.group(2).strip()
            info["return_type"] = func_match.group(3).strip()
        
        # Extract specification
        spec_match = re.search(r'-- << SPEC START >>(.*?)-- << SPEC END >>', task_lean_code, re.DOTALL)
        if spec_match:
            info["specification"] = spec_match.group(1).strip()
        
        # Analyze problem type
        if "Array" in task_lean_code:
            info["problem_type"] = "array_processing"
        elif "Bool" in info["return_type"]:
            info["problem_type"] = "boolean_logic"
        elif any(word in problem_description.lower() for word in ["minimum", "maximum", "compare"]):
            info["problem_type"] = "comparison"
        else:
            info["problem_type"] = "arithmetic"
        
        return info
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"approach": "direct", "proof_strategy": "simp", "insights": "", "pitfalls": ""}
    
    def record_attempt(self, attempt: Solution):
        """Record failed attempt for learning"""
        self.previous_attempts.append(attempt)


class GenerationAgent:
    """Agent 2: Generates Lean 4 code and proofs"""
    
    def __init__(self):
        self.agent = LLM_Agent(model="gpt-4o")
        profiler.start_timer("Generation.rag_init")
        self.rag_system = self._initialize_rag()
        profiler.end_timer("Generation.rag_init")
    
    def _initialize_rag(self) -> Optional[Dict]:
        """Initialize RAG system"""
        try:
            if os.path.exists("embeddings.npy") and os.path.exists("embeddings_chunks.pkl"):
                return {
                    "embedding_model": OpenAIEmbeddingModel(),
                    "database_file": "embeddings.npy"
                }
        except Exception as e:
            print(f"[Generation Agent] RAG init failed: {e}")
        return None
    
    @timer_decorator("GenerationAgent.query_rag")
    def _query_rag(self, query: str, k: int = 3) -> str:
        """Query RAG for relevant knowledge"""
        if not self.rag_system:
            return ""
        
        try:
            chunks, scores = VectorDB.get_top_k(
                self.rag_system["database_file"],
                self.rag_system["embedding_model"],
                query,
                k=k
            )
            return "\n".join(chunks) if chunks else ""
        except Exception as e:
            print(f"[Generation Agent] RAG query failed: {e}")
            return ""
    
    @timer_decorator("GenerationAgent.generate_implementation")
    def generate_implementation(self, plan: Dict[str, Any]) -> str:
        """Generate implementation based on strategic plan"""
        
        profiler.start_timer("Generation.rag_query_code")
        # Query RAG for relevant examples
        rag_query = f"Lean 4 {plan['function_name']} {plan['problem_type']} implementation"
        rag_context = self._query_rag(rag_query)
        profiler.end_timer("Generation.rag_query_code")
        
        profiler.start_timer("Generation.llm_call_code")
        # Create generation prompt
        prompt = f"""
        Generate Lean 4 implementation following this strategic plan:

        FUNCTION: {plan['function_name']} {plan['parameters']} : {plan['return_type']}
        PROBLEM TYPE: {plan['problem_type']}
        SPECIFICATION: {plan['specification']}
        
        STRATEGIC APPROACH: {plan.get('approach', 'direct')}
        KEY INSIGHTS: {plan.get('insights', '')}
        PITFALLS TO AVOID: {plan.get('pitfalls', '')}

        RELEVANT KNOWLEDGE:
        {rag_context}

        REQUIREMENTS:
        1. Generate ONLY the implementation body (no def signature)
        2. Use appropriate Lean 4 syntax and library functions
        3. Handle edge cases based on specification
        4. Follow the strategic approach outlined above

        Implementation:
        """
        
        try:
            response = self.agent.get_response([{"role": "user", "content": prompt}])
            profiler.end_timer("Generation.llm_call_code")
            
            profiler.start_timer("Generation.clean_code")
            result = self._clean_code_response(response)
            profiler.end_timer("Generation.clean_code")
            return result
        except Exception as e:
            profiler.end_timer("Generation.llm_call_code")
            print(f"[Generation Agent] Code generation failed: {e}")
            return "sorry"
    
    @timer_decorator("GenerationAgent.generate_proof")
    def generate_proof(self, plan: Dict[str, Any], code: str) -> str:
        """Generate proof based on strategic plan and implementation"""
        
        profiler.start_timer("Generation.rag_query_proof")
        # Query RAG for proof patterns
        rag_query = f"Lean 4 proof {plan['problem_type']} {plan.get('proof_strategy', 'simp')}"
        rag_context = self._query_rag(rag_query)
        profiler.end_timer("Generation.rag_query_proof")
        
        profiler.start_timer("Generation.llm_call_proof")
        # Create proof generation prompt
        prompt = f"""
        Generate Lean 4 proof tactics following this strategic plan:

        FUNCTION: {plan['function_name']}
        IMPLEMENTATION: {code}
        SPECIFICATION: {plan['specification']}
        
        PROOF STRATEGY: {plan.get('proof_strategy', 'simp')}
        KEY INSIGHTS: {plan.get('insights', '')}

        RELEVANT PROOF PATTERNS:
        {rag_context}

        REQUIREMENTS:
        1. Generate proof tactics that work with the implementation
        2. Handle specification structure (∧, ∨, ∀, ∃, etc.)
        3. Use appropriate tactics: constructor, simp, cases, exact, etc.
        4. Start with unfold {plan['function_name']} {plan['function_name']}_spec

        Proof tactics:
        """
        
        try:
            response = self.agent.get_response([{"role": "user", "content": prompt}])
            profiler.end_timer("Generation.llm_call_proof")
            
            profiler.start_timer("Generation.clean_proof")
            result = self._clean_proof_response(response)
            profiler.end_timer("Generation.clean_proof")
            return result
        except Exception as e:
            profiler.end_timer("Generation.llm_call_proof")
            print(f"[Generation Agent] Proof generation failed: {e}")
            return "sorry"
    
    def _clean_code_response(self, response: str) -> str:
        """Clean code from LLM response"""
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        lines = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('def')]
        return lines[0] if lines else "sorry"
    
    def _clean_proof_response(self, response: str) -> str:
        """Clean proof from LLM response"""
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        response = re.sub(r'^\s*by\s+', '', response.strip())
        lines = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.strip().startswith('--')]
        return '\n'.join(lines) if lines else "sorry"


class VerificationAgent:
    """Agent 3: Verifies solutions and provides feedback"""
    
    def __init__(self):
        self.agent = LLM_Agent(model="gpt-4o")
        profiler.start_timer("Verification.rag_init")
        self.rag_system = self._initialize_rag()
        profiler.end_timer("Verification.rag_init")
    
    def _initialize_rag(self) -> Optional[Dict]:
        """Initialize RAG system for error patterns"""
        try:
            if os.path.exists("embeddings.npy") and os.path.exists("embeddings_chunks.pkl"):
                return {
                    "embedding_model": OpenAIEmbeddingModel(),
                    "database_file": "embeddings.npy"
                }
        except:
            pass
        return None
    
    @timer_decorator("VerificationAgent.verify_solution")
    def verify_solution(self, solution: Solution, task_lean_code: str) -> bool:
        """Verify solution by executing Lean code"""
        try:
            profiler.start_timer("Verification.prepare_code")
            test_code = task_lean_code.replace("{{code}}", solution.code)
            test_code = test_code.replace("{{proof}}", solution.proof)
            profiler.end_timer("Verification.prepare_code")
            
            profiler.start_timer("Verification.lean_execution")
            result = execute_lean_code(test_code)
            profiler.end_timer("Verification.lean_execution")
            
            if "executed successfully" in result or "No errors found" in result:
                solution.success = True
                solution.reasoning_trace.append("Verification: PASS")
                return True
            else:
                error_msg = result.split("Lean Error:")[-1].strip() if "Lean Error:" in result else result
                solution.error = error_msg
                solution.reasoning_trace.append(f"Verification: FAIL - {error_msg[:100]}...")
                return False
                
        except Exception as e:
            solution.error = str(e)
            solution.reasoning_trace.append(f"Verification error: {str(e)}")
            return False
    
    @timer_decorator("VerificationAgent.analyze_error")
    def analyze_error_and_suggest_fix(self, solution: Solution, plan: Dict[str, Any]) -> Dict[str, str]:
        """Analyze error and suggest corrections"""
        
        if not solution.error:
            return {"suggestion": "no_error", "approach": "keep_current"}
        
        profiler.start_timer("Verification.rag_query_error")
        # Query RAG for error patterns
        if self.rag_system:
            try:
                rag_query = f"Lean 4 error fix {solution.error[:50]}"
                chunks, _ = VectorDB.get_top_k(
                    self.rag_system["database_file"],
                    self.rag_system["embedding_model"],
                    rag_query,
                    k=2
                )
                rag_context = "\n".join(chunks) if chunks else ""
            except:
                rag_context = ""
        else:
            rag_context = ""
        profiler.end_timer("Verification.rag_query_error")
        
        profiler.start_timer("Verification.llm_call_error")
        # Create error analysis prompt
        prompt = f"""
        Analyze this Lean 4 error and suggest a fix:

        FUNCTION: {plan['function_name']}
        CURRENT CODE: {solution.code}
        CURRENT PROOF: {solution.proof}
        ERROR MESSAGE: {solution.error}
        
        CONTEXT FROM KNOWLEDGE BASE:
        {rag_context}

        Suggest specific fixes:
        1. What is the root cause of this error?
        2. Should we modify the code or proof?
        3. What specific changes are needed?

        Respond in JSON format with keys: root_cause, fix_target, specific_changes
        """
        
        try:
            response = self.agent.get_response([{"role": "user", "content": prompt}])
            profiler.end_timer("Verification.llm_call_error")
            
            # Parse JSON response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            profiler.end_timer("Verification.llm_call_error")
            pass
        
        return {
            "root_cause": "unknown",
            "fix_target": "proof", 
            "specific_changes": "try different tactics"
        }


class MultiAgentLeanSolver:
    """Main multi-agent orchestrator with performance profiling"""
    
    def __init__(self):
        profiler.start_timer("Solver.initialization")
        self.planning_agent = PlanningAgent()
        self.generation_agent = GenerationAgent()
        self.verification_agent = VerificationAgent()
        profiler.end_timer("Solver.initialization")
        
        print("[Multi-Agent Solver] Initialized 3-agent architecture:")
        print("  1. Planning Agent (o3-mini) - Strategy & Task Decomposition")
        print("  2. Generation Agent (gpt-4o) - Code & Proof Generation with RAG")
        print("  3. Verification Agent (gpt-4o) - Verification & Feedback")
    
    @timer_decorator("MultiAgentSolver.solve")
    def solve_with_multi_agent_workflow(self, problem_description: str, task_lean_code: str, max_iterations: int = 3) -> Solution:
        """Multi-agent workflow with corrective feedback and performance profiling"""
        
        print("[Multi-Agent Solver] Starting multi-agent workflow...")
        
        # Step 1: Strategic Planning
        print("[Step 1] Planning Agent: Analyzing problem and creating strategy...")
        plan = self.planning_agent.analyze_problem(problem_description, task_lean_code)
        print(f"[Step 1] Strategy: {plan.get('approach', 'direct')} approach")
        
        # Step 2: Iterative Generation and Verification
        for iteration in range(max_iterations):
            print(f"[Step 2.{iteration+1}] Generation-Verification Cycle {iteration+1}/{max_iterations}")
            
            # Generate implementation
            print("[Generation Agent] Generating implementation...")
            code = self.generation_agent.generate_implementation(plan)
            print(f"[Generation Agent] Generated code: {code[:50]}...")
            
            # Generate proof
            print("[Generation Agent] Generating proof...")
            proof = self.generation_agent.generate_proof(plan, code)
            print(f"[Generation Agent] Generated proof: {proof[:50]}...")
            
            # Create solution
            solution = Solution(
                code=code,
                proof=proof,
                approach=f"multi_agent_{plan.get('approach', 'direct')}",
                reasoning_trace=[
                    f"Planning: {plan.get('approach', 'direct')} strategy",
                    f"Generation iteration: {iteration+1}",
                    f"Code: {code[:30]}...",
                    f"Proof strategy: {plan.get('proof_strategy', 'simp')}"
                ]
            )
            
            # Verify solution
            print("[Verification Agent] Verifying solution...")
            if self.verification_agent.verify_solution(solution, task_lean_code):
                print(f"[Multi-Agent Solver] SUCCESS after {iteration+1} iterations!")
                solution.reasoning_trace.append(f"Verified successfully on iteration {iteration+1}")
                return solution
            
            # If verification failed and we have more iterations
            if iteration < max_iterations - 1:
                print("[Verification Agent] Solution failed, analyzing error...")
                error_analysis = self.verification_agent.analyze_error_and_suggest_fix(solution, plan)
                print(f"[Verification Agent] Error cause: {error_analysis.get('root_cause', 'unknown')}")
                
                # Record failed attempt for learning
                self.planning_agent.record_attempt(solution)
                
                # Update plan based on feedback
                plan["previous_error"] = solution.error
                plan["error_analysis"] = error_analysis
                print("[Planning Agent] Updating strategy based on feedback...")
        
        # All iterations failed
        print("[Multi-Agent Solver] All iterations failed")
        solution.reasoning_trace.append(f"Failed after {max_iterations} multi-agent iterations")
        return solution


def main_workflow(problem_description: str, task_lean_code: str = "") -> Dict[str, str]:
    """
    Multi-agent workflow with comprehensive performance profiling.
    """
    print("[Main Workflow] Starting Lab 2 Multi-Agent Lean 4 Theorem Prover...")
    
    # Reset profiler for this task
    global profiler
    profiler = PerformanceProfiler()
    
    profiler.start_timer("Total.workflow")
    
    # Input validation
    if not task_lean_code or not problem_description:
        print("[Main Workflow] Missing required inputs")
        return {"code": "sorry", "proof": "sorry"}
    
    # Initialize multi-agent system
    solver = MultiAgentLeanSolver()
    
    try:
        # Execute multi-agent workflow
        solution = solver.solve_with_multi_agent_workflow(problem_description, task_lean_code)
        
        profiler.end_timer("Total.workflow")
        
        # Print performance summary
        print(profiler.get_summary())
        
        # Log results
        print(f"[Main Workflow] Final approach: {solution.approach}")
        print(f"[Main Workflow] Success: {solution.success}")
        print(f"[Main Workflow] Reasoning trace: {' → '.join(solution.reasoning_trace[-2:])}")
        
        return {
            "code": solution.code,
            "proof": solution.proof
        }
        
    except Exception as e:
        profiler.end_timer("Total.workflow")
        print(f"[Main Workflow] Unexpected error: {e}")
        print(profiler.get_summary())
        return {"code": "sorry", "proof": "sorry"}

def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    """Read problem description and Lean code template from task directory."""
    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()
    return problem_description, lean_code_template


def get_unit_tests_from_taskpath(task_path: str) -> str:
    """Read unit tests from task directory."""
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    return unit_tests


def get_task_lean_template_from_taskpath(task_path: str) -> str:
    """Read Lean code template from task directory."""
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    return task_lean_template