import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time

from src.agents import Reasoning_Agent, LLM_Agent
from src.lean_runner import execute_lean_code
from src.embedding_db import VectorDB
from src.embedding_models import OpenAIEmbeddingModel


class ApproachType(Enum):
    """Enumeration of different reasoning approaches for Lean 4 problems."""
    DIRECT = "direct"
    SPECIFICATION_DRIVEN = "specification_driven"
    LIBRARY_HEAVY = "library_heavy"
    CONSTRUCTIVE = "constructive"


@dataclass
class ApproachScore:
    """Scoring metrics for different approaches."""
    feasibility: int  # 1-5 scale
    robustness: int   # 1-5 scale
    proof_complexity: int  # 1-5 scale (lower is better)
    implementation_time: int  # 1-5 scale (lower is faster)
    error_recovery: int  # 1-5 scale
    
    def calculate_priority(self) -> float:
        """Calculate priority score for approach selection."""
        return (self.feasibility * self.robustness) - self.proof_complexity + (self.error_recovery * 0.5)


@dataclass
class ApproachPlan:
    """Detailed plan for a specific approach."""
    approach_type: ApproachType
    score: ApproachScore
    strategy: str
    implementation_notes: str
    proof_tactics: List[str]
    risk_factors: List[str]
    fallback_plan: str


class TreeOfThoughtsLeanAgent:
    """
    Tree of Thoughts implementation for Lean 4 theorem proving.
    Implements multiple reasoning paths and systematic approach selection.
    """
    
    def __init__(self):
        """Initialize the ToT agent with necessary components."""
        self.planning_agent = Reasoning_Agent(model="o3-mini")
        self.generation_agent = LLM_Agent(model="gpt-4o")
        self.verification_agent = LLM_Agent(model="gpt-4o")
        
        # Initialize RAG system if embeddings exist
        self.rag_system = None
        self._initialize_rag()
        
        # Conversation history for context
        self.conversation_history: List[Dict[str, Any]] = []
        self.failed_attempts: List[Dict[str, str]] = []
    
    def _initialize_rag(self) -> None:
        """Initialize RAG system if embedding database exists."""
        try:
            if os.path.exists("embeddings.npy") and os.path.exists("embeddings_chunks.pkl"):
                self.rag_system = {
                    "embedding_model": OpenAIEmbeddingModel(),
                    "database_file": "embeddings.npy"
                }
                print("[ToT Agent] RAG system initialized successfully")
        except Exception as e:
            print(f"[ToT Agent] Could not initialize RAG system: {e}")
    
    def _query_rag(self, query: str, k: int = 3) -> List[str]:
        """Query RAG system for relevant examples."""
        if not self.rag_system:
            return []
        
        try:
            chunks, scores = VectorDB.get_top_k(
                self.rag_system["database_file"],
                self.rag_system["embedding_model"],
                query,
                k=k
            )
            return chunks
        except Exception as e:
            print(f"[ToT Agent] RAG query failed: {e}")
            return []
    
    def generate_approach_plans(self, problem_description: str, task_lean_code: str) -> List[ApproachPlan]:
        """
        Generate multiple approach plans using Tree of Thoughts methodology.
        """
        print("[ToT Agent] Generating approach plans...")
        
        # Extract key information
        function_name = self._extract_function_name(task_lean_code)
        specification = self._extract_specification(task_lean_code)
        
        # Query RAG for similar problems
        rag_examples = self._query_rag(f"{function_name} {problem_description}")
        rag_context = "\n".join(rag_examples[:2]) if rag_examples else ""
        
        approaches = []
        
        # Path 1: Direct Implementation
        direct_plan = self._generate_direct_approach(problem_description, specification, rag_context)
        approaches.append(direct_plan)
        
        # Path 2: Specification-Driven
        spec_driven_plan = self._generate_specification_driven_approach(problem_description, specification, rag_context)
        approaches.append(spec_driven_plan)
        
        # Path 3: Library-Heavy
        library_plan = self._generate_library_heavy_approach(problem_description, specification, rag_context)
        approaches.append(library_plan)
        
        # Path 4: Constructive Proof
        constructive_plan = self._generate_constructive_approach(problem_description, specification, rag_context)
        approaches.append(constructive_plan)
        
        return approaches
    
    def _generate_direct_approach(self, problem_desc: str, spec: str, rag_context: str) -> ApproachPlan:
        """Generate direct implementation approach plan."""
        
        planning_prompt = f"""
        Analyze this Lean 4 problem for a DIRECT IMPLEMENTATION approach:
        
        Problem: {problem_desc}
        Specification: {spec}
        Similar Examples: {rag_context}
        
        For the DIRECT approach:
        1. Implement the most straightforward algorithm
        2. Focus on correctness over optimization
        3. Use basic proof tactics (simp, rw, cases)
        4. Rate this approach on feasibility (1-5), robustness (1-5), proof_complexity (1-5), implementation_time (1-5), error_recovery (1-5)
        
        Respond in JSON format:
        {{
            "strategy": "brief strategy description",
            "implementation_notes": "key implementation decisions",
            "proof_tactics": ["list", "of", "tactics"],
            "risk_factors": ["potential", "issues"],
            "fallback_plan": "backup approach",
            "feasibility": 4,
            "robustness": 3,
            "proof_complexity": 2,
            "implementation_time": 2,
            "error_recovery": 3
        }}
        """
        
        response = self.planning_agent.get_response([{"role": "user", "content": planning_prompt}])
        plan_data = self._parse_json_response(response)
        
        score = ApproachScore(
            feasibility=plan_data.get("feasibility", 4),
            robustness=plan_data.get("robustness", 3),
            proof_complexity=plan_data.get("proof_complexity", 2),
            implementation_time=plan_data.get("implementation_time", 2),
            error_recovery=plan_data.get("error_recovery", 3)
        )
        
        return ApproachPlan(
            approach_type=ApproachType.DIRECT,
            score=score,
            strategy=plan_data.get("strategy", "Direct straightforward implementation"),
            implementation_notes=plan_data.get("implementation_notes", ""),
            proof_tactics=plan_data.get("proof_tactics", ["simp", "rw", "cases"]),
            risk_factors=plan_data.get("risk_factors", []),
            fallback_plan=plan_data.get("fallback_plan", "")
        )
    
    def _generate_specification_driven_approach(self, problem_desc: str, spec: str, rag_context: str) -> ApproachPlan:
        """Generate specification-driven approach plan."""
        
        planning_prompt = f"""
        Analyze this Lean 4 problem for a SPECIFICATION-DRIVEN approach:
        
        Problem: {problem_desc}
        Specification: {spec}
        Similar Examples: {rag_context}
        
        For the SPECIFICATION-DRIVEN approach:
        1. Work backwards from the specification
        2. Design implementation to make proof easier
        3. Align data structures with proof structure
        4. Rate this approach on feasibility (1-5), robustness (1-5), proof_complexity (1-5), implementation_time (1-5), error_recovery (1-5)
        
        Respond in JSON format:
        {{
            "strategy": "brief strategy description",
            "implementation_notes": "key implementation decisions",
            "proof_tactics": ["list", "of", "tactics"],
            "risk_factors": ["potential", "issues"],
            "fallback_plan": "backup approach",
            "feasibility": 3,
            "robustness": 4,
            "proof_complexity": 3,
            "implementation_time": 3,
            "error_recovery": 4
        }}
        """
        
        response = self.planning_agent.get_response([{"role": "user", "content": planning_prompt}])
        plan_data = self._parse_json_response(response)
        
        score = ApproachScore(
            feasibility=plan_data.get("feasibility", 3),
            robustness=plan_data.get("robustness", 4),
            proof_complexity=plan_data.get("proof_complexity", 3),
            implementation_time=plan_data.get("implementation_time", 3),
            error_recovery=plan_data.get("error_recovery", 4)
        )
        
        return ApproachPlan(
            approach_type=ApproachType.SPECIFICATION_DRIVEN,
            score=score,
            strategy=plan_data.get("strategy", "Work backwards from specification"),
            implementation_notes=plan_data.get("implementation_notes", ""),
            proof_tactics=plan_data.get("proof_tactics", ["unfold", "simp", "constructor"]),
            risk_factors=plan_data.get("risk_factors", []),
            fallback_plan=plan_data.get("fallback_plan", "")
        )
    
    def _generate_library_heavy_approach(self, problem_desc: str, spec: str, rag_context: str) -> ApproachPlan:
        """Generate library-heavy approach plan."""
        
        planning_prompt = f"""
        Analyze this Lean 4 problem for a LIBRARY-HEAVY approach:
        
        Problem: {problem_desc}
        Specification: {spec}
        Similar Examples: {rag_context}
        
        For the LIBRARY-HEAVY approach:
        1. Leverage Mathlib extensively
        2. Build using proven library components
        3. Reuse established proof patterns
        4. Rate this approach on feasibility (1-5), robustness (1-5), proof_complexity (1-5), implementation_time (1-5), error_recovery (1-5)
        
        Respond in JSON format:
        {{
            "strategy": "brief strategy description",
            "implementation_notes": "key implementation decisions",
            "proof_tactics": ["list", "of", "tactics"],
            "risk_factors": ["potential", "issues"],
            "fallback_plan": "backup approach",
            "feasibility": 4,
            "robustness": 5,
            "proof_complexity": 2,
            "implementation_time": 4,
            "error_recovery": 5
        }}
        """
        
        response = self.planning_agent.get_response([{"role": "user", "content": planning_prompt}])
        plan_data = self._parse_json_response(response)
        
        score = ApproachScore(
            feasibility=plan_data.get("feasibility", 4),
            robustness=plan_data.get("robustness", 5),
            proof_complexity=plan_data.get("proof_complexity", 2),
            implementation_time=plan_data.get("implementation_time", 4),
            error_recovery=plan_data.get("error_recovery", 5)
        )
        
        return ApproachPlan(
            approach_type=ApproachType.LIBRARY_HEAVY,
            score=score,
            strategy=plan_data.get("strategy", "Use Mathlib extensively"),
            implementation_notes=plan_data.get("implementation_notes", ""),
            proof_tactics=plan_data.get("proof_tactics", ["exact", "apply", "rw"]),
            risk_factors=plan_data.get("risk_factors", []),
            fallback_plan=plan_data.get("fallback_plan", "")
        )
    
    def _generate_constructive_approach(self, problem_desc: str, spec: str, rag_context: str) -> ApproachPlan:
        """Generate constructive proof approach plan."""
        
        planning_prompt = f"""
        Analyze this Lean 4 problem for a CONSTRUCTIVE PROOF approach:
        
        Problem: {problem_desc}
        Specification: {spec}
        Similar Examples: {rag_context}
        
        For the CONSTRUCTIVE approach:
        1. Make the proof constructively obvious
        2. Use dependent types or proof-carrying patterns
        3. Make invariants explicit in code structure
        4. Rate this approach on feasibility (1-5), robustness (1-5), proof_complexity (1-5), implementation_time (1-5), error_recovery (1-5)
        
        Respond in JSON format:
        {{
            "strategy": "brief strategy description",
            "implementation_notes": "key implementation decisions",
            "proof_tactics": ["list", "of", "tactics"],
            "risk_factors": ["potential", "issues"],
            "fallback_plan": "backup approach",
            "feasibility": 3,
            "robustness": 4,
            "proof_complexity": 4,
            "implementation_time": 4,
            "error_recovery": 3
        }}
        """
        
        response = self.planning_agent.get_response([{"role": "user", "content": planning_prompt}])
        plan_data = self._parse_json_response(response)
        
        score = ApproachScore(
            feasibility=plan_data.get("feasibility", 3),
            robustness=plan_data.get("robustness", 4),
            proof_complexity=plan_data.get("proof_complexity", 4),
            implementation_time=plan_data.get("implementation_time", 4),
            error_recovery=plan_data.get("error_recovery", 3)
        )
        
        return ApproachPlan(
            approach_type=ApproachType.CONSTRUCTIVE,
            score=score,
            strategy=plan_data.get("strategy", "Make correctness self-evident"),
            implementation_notes=plan_data.get("implementation_notes", ""),
            proof_tactics=plan_data.get("proof_tactics", ["constructor", "exact", "assumption"]),
            risk_factors=plan_data.get("risk_factors", []),
            fallback_plan=plan_data.get("fallback_plan", "")
        )
    
    def select_best_approach(self, approaches: List[ApproachPlan]) -> ApproachPlan:
        """Select the best approach based on priority scoring."""
        print("[ToT Agent] Evaluating and selecting best approach...")
        
        best_approach = max(approaches, key=lambda x: x.score.calculate_priority())
        
        print(f"[ToT Agent] Selected approach: {best_approach.approach_type.value}")
        print(f"[ToT Agent] Priority score: {best_approach.score.calculate_priority():.2f}")
        
        return best_approach
    
    def implement_approach(self, approach: ApproachPlan, problem_description: str, task_lean_code: str) -> Dict[str, str]:
        """
        Implement the selected approach with iterative refinement.
        """
        print(f"[ToT Agent] Implementing {approach.approach_type.value} approach...")
        
        # Extract context
        function_name = self._extract_function_name(task_lean_code)
        specification = self._extract_specification(task_lean_code)
        
        # Query RAG for implementation examples
        rag_examples = self._query_rag(f"implementation {function_name} {approach.approach_type.value}")
        rag_context = "\n".join(rag_examples) if rag_examples else ""
        
        max_attempts = 3
        for attempt in range(max_attempts):
            print(f"[ToT Agent] Implementation attempt {attempt + 1}/{max_attempts}")
            
            try:
                # Generate implementation
                code = self._generate_code(approach, problem_description, specification, rag_context, attempt)
                
                # Generate proof
                proof = self._generate_proof(approach, problem_description, specification, code, rag_context, attempt)
                
                # Verify the solution
                verification_result = self._verify_solution(task_lean_code, code, proof)
                
                if verification_result["success"]:
                    print("[ToT Agent] Solution verified successfully!")
                    return {"code": code, "proof": proof}
                else:
                    print(f"[ToT Agent] Verification failed: {verification_result['error']}")
                    self.failed_attempts.append({
                        "code": code,
                        "proof": proof,
                        "error": verification_result["error"],
                        "attempt": attempt + 1
                    })
            
            except Exception as e:
                print(f"[ToT Agent] Implementation attempt {attempt + 1} failed: {e}")
                self.failed_attempts.append({
                    "error": str(e),
                    "attempt": attempt + 1
                })
        
        print(f"[ToT Agent] All attempts failed for {approach.approach_type.value} approach")
        return {"code": "sorry", "proof": "sorry"}
    
    def _generate_code(self, approach: ApproachPlan, problem_desc: str, spec: str, rag_context: str, attempt: int) -> str:
        """Generate code implementation based on the selected approach."""
        
        # Build context from failed attempts
        failure_context = ""
        if self.failed_attempts:
            recent_failures = self.failed_attempts[-2:]  # Last 2 failures
            failure_context = "\n\nPrevious failed attempts:\n"
            for i, failure in enumerate(recent_failures):
                failure_context += f"Attempt {failure.get('attempt', i+1)}: {failure.get('error', 'Unknown error')}\n"
        
        generation_prompt = f"""
        Generate Lean 4 code implementation using the {approach.approach_type.value} approach.
        
        Problem: {problem_desc}
        Specification: {spec}
        Strategy: {approach.strategy}
        Implementation Notes: {approach.implementation_notes}
        Similar Examples: {rag_context}
        {failure_context}
        
        Requirements:
        1. Generate ONLY the implementation code (no function signature or comments)
        2. Follow the {approach.approach_type.value} strategy
        3. Ensure the implementation satisfies the specification
        4. Avoid using 'sorry' in the implementation
        5. Handle edge cases appropriately
        
        Attempt: {attempt + 1}
        """
        
        response = self.generation_agent.get_response([{"role": "user", "content": generation_prompt}])
        return self._clean_code_response(response)
    
    def _generate_proof(self, approach: ApproachPlan, problem_desc: str, spec: str, code: str, rag_context: str, attempt: int) -> str:
        """Generate proof based on the selected approach and implementation."""
        
        # Build context from failed attempts
        failure_context = ""
        if self.failed_attempts:
            recent_failures = self.failed_attempts[-2:]
            failure_context = "\n\nPrevious failed proof attempts:\n"
            for i, failure in enumerate(recent_failures):
                if "proof" in failure:
                    failure_context += f"Failed proof {failure.get('attempt', i+1)}: {failure.get('error', 'Unknown error')}\n"
        
        proof_prompt = f"""
        Generate Lean 4 proof using the {approach.approach_type.value} approach.
        
        Problem: {problem_desc}
        Specification: {spec}
        Implementation: {code}
        Recommended Tactics: {', '.join(approach.proof_tactics)}
        Similar Examples: {rag_context}
        {failure_context}
        
        Requirements:
        1. Generate ONLY the proof tactics (no 'by' keyword or comments)
        2. Use tactics from: {', '.join(approach.proof_tactics)}
        3. The proof should be complete and valid
        4. Avoid using 'sorry' in the proof
        5. Handle all cases in the specification
        
        Start with 'unfold' tactics if needed, then proceed with the proof.
        Attempt: {attempt + 1}
        """
        
        response = self.generation_agent.get_response([{"role": "user", "content": proof_prompt}])
        return self._clean_proof_response(response)
    
    def _verify_solution(self, task_lean_code: str, code: str, proof: str) -> Dict[str, Any]:
        """Verify the generated solution by executing Lean code."""
        
        # Replace placeholders with generated solution
        test_code = task_lean_code.replace("{{code}}", code).replace("{{proof}}", proof)
        
        # Execute the code
        result = execute_lean_code(test_code)
        
        if "executed successfully" in result:
            return {"success": True, "output": result}
        else:
            error_msg = result.split("Lean Error:")[-1].strip() if "Lean Error:" in result else result
            return {"success": False, "error": error_msg}
    
    def _extract_function_name(self, task_lean_code: str) -> str:
        """Extract function name from Lean code template."""
        match = re.search(r'def\s+(\w+)', task_lean_code)
        return match.group(1) if match else "unknown_function"
    
    def _extract_specification(self, task_lean_code: str) -> str:
        """Extract specification from Lean code template."""
        spec_match = re.search(r'-- << SPEC START >>(.*?)-- << SPEC END >>', task_lean_code, re.DOTALL)
        return spec_match.group(1).strip() if spec_match else ""
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM, with fallback handling."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except (json.JSONDecodeError, AttributeError):
            print(f"[ToT Agent] Failed to parse JSON response: {response[:100]}...")
            return {}
    
    def _clean_code_response(self, response: str) -> str:
        """Clean and extract code from LLM response."""
        # Remove markdown code blocks
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Remove comments and extra whitespace
        lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('--')]
        
        return '\n'.join(lines) if lines else response.strip()
    
    def _clean_proof_response(self, response: str) -> str:
        """Clean and extract proof from LLM response."""
        # Remove markdown code blocks
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        # Remove 'by' keyword if present at the start
        response = re.sub(r'^\s*by\s+', '', response.strip())
        
        # Clean up the proof
        lines = [line.strip() for line in response.split('\n') if line.strip() and not line.strip().startswith('--')]
        
        return '\n'.join(lines) if lines else response.strip()


def main_workflow(problem_description: str, task_lean_code: str = "") -> Dict[str, str]:
    """
    Main workflow implementing Tree of Thoughts for Lean 4 theorem proving.
    
    Args:
        problem_description: Natural language problem description
        task_lean_code: Lean code template with placeholders
    
    Returns:
        Dictionary with "code" and "proof" keys containing the generated solution
    """
    print("[Main Workflow] Starting Tree of Thoughts Lean 4 Agent...")
    
    # Initialize ToT agent
    tot_agent = TreeOfThoughtsLeanAgent()
    
    try:
        # Phase 1: Generate multiple approach plans
        approaches = tot_agent.generate_approach_plans(problem_description, task_lean_code)
        
        # Phase 2: Select best approach
        selected_approach = tot_agent.select_best_approach(approaches)
        
        # Phase 3: Implement selected approach
        solution = tot_agent.implement_approach(selected_approach, problem_description, task_lean_code)
        
        # If primary approach fails, try fallback approaches
        if solution["code"] == "sorry" and len(approaches) > 1:
            print("[Main Workflow] Primary approach failed, trying fallback approaches...")
            
            # Sort remaining approaches by priority
            remaining_approaches = [a for a in approaches if a != selected_approach]
            remaining_approaches.sort(key=lambda x: x.score.calculate_priority(), reverse=True)
            
            for fallback_approach in remaining_approaches:
                print(f"[Main Workflow] Trying fallback: {fallback_approach.approach_type.value}")
                solution = tot_agent.implement_approach(fallback_approach, problem_description, task_lean_code)
                
                if solution["code"] != "sorry":
                    break
        
        print(f"[Main Workflow] Final solution - Code: {len(solution['code'])} chars, Proof: {len(solution['proof'])} chars")
        return solution
    
    except Exception as e:
        print(f"[Main Workflow] Unexpected error: {e}")
        return {"code": "sorry", "proof": "sorry"}


# Keep existing utility functions
def get_problem_and_code_from_taskpath(task_path: str) -> Tuple[str, str]:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and also read the file 
    that contains the task description, which is "description.txt".
    
    After reading the files, it will return a tuple of the problem description and the Lean code template.
    
    Args:
        task_path: Path to the task file
    """
    problem_description = ""
    lean_code_template = ""
    
    with open(os.path.join(task_path, "description.txt"), "r") as f:
        problem_description = f.read()

    with open(os.path.join(task_path, "task.lean"), "r") as f:
        lean_code_template = f.read()

    return problem_description, lean_code_template


def get_unit_tests_from_taskpath(task_path: str) -> str:
    """
    Reads a directory in the format of task_id_*. It will read the file "tests.lean" and return the unit tests.
    """
    with open(os.path.join(task_path, "tests.lean"), "r") as f:
        unit_tests = f.read()
    
    return unit_tests


def get_task_lean_template_from_taskpath(task_path: str) -> str:
    """
    Reads a directory in the format of task_id_*. It will read the file "task.lean" and return the Lean code template.
    """
    with open(os.path.join(task_path, "task.lean"), "r") as f:
        task_lean_template = f.read()
    
    return task_lean_template