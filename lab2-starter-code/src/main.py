import os
import re
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from src.agents import Reasoning_Agent, LLM_Agent
from src.lean_runner import execute_lean_code
from src.embedding_db import VectorDB
from src.embedding_models import OpenAIEmbeddingModel


class ReasoningPath(Enum):
    """Different reasoning paths for Tree of Thoughts"""
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


class TreeOfThoughtsAgent:
    """Tree of Thoughts implementation for Lean 4 theorem proving"""
    
    def __init__(self):
        self.planning_agent = Reasoning_Agent(model="o3-mini")
        self.generation_agent = LLM_Agent(model="gpt-4o")
        self.verification_agent = LLM_Agent(model="gpt-4o")
        
        # Initialize RAG system
        self.rag_system = None
        self._initialize_rag()
        
        # Conversation history for learning
        self.conversation_history: List[Dict[str, Any]] = []
        self.failed_attempts: List[Solution] = []
    
    def _initialize_rag(self) -> None:
        """Initialize RAG system for knowledge retrieval"""
        try:
            if os.path.exists("embeddings.npy") and os.path.exists("embeddings_chunks.pkl"):
                self.rag_system = {
                    "embedding_model": OpenAIEmbeddingModel(),
                    "database_file": "embeddings.npy"
                }
                print("[ToT Agent] RAG system initialized")
        except Exception as e:
            print(f"[ToT Agent] RAG initialization failed: {e}")
    
    def _query_rag(self, query: str, k: int = 5) -> str:
        """Query RAG system for relevant knowledge"""
        if not self.rag_system:
            return ""
        
        try:
            chunks, scores = VectorDB.get_top_k(
                self.rag_system["database_file"],
                self.rag_system["embedding_model"],
                query,
                k=k
            )
            return "\n".join(chunks[:3]) if chunks else ""
        except Exception as e:
            print(f"[ToT Agent] RAG query failed: {e}")
            return ""
    
    def extract_problem_info(self, problem_description: str, task_lean_code: str) -> Dict[str, str]:
        """Extract and analyze problem information"""
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
            info["complexity"] = "high"
        elif "Bool" in info["return_type"]:
            info["problem_type"] = "boolean_logic"
            info["complexity"] = "medium"
        elif any(word in problem_description.lower() for word in ["minimum", "maximum", "compare"]):
            info["problem_type"] = "comparison"
            info["complexity"] = "medium"
        else:
            info["problem_type"] = "arithmetic"
            info["complexity"] = "low"
        
        return info
    
    def generate_thought_nodes(self, problem_info: Dict[str, str], rag_context: str) -> List[ThoughtNode]:
        """Generate multiple reasoning paths using Tree of Thoughts"""
        print("[ToT Agent] Generating thought nodes...")
        
        # Build context for planning
        planning_context = f"""
Problem Analysis:
- Function: {problem_info['function_name']} {problem_info['parameters']} : {problem_info['return_type']}
- Type: {problem_info['problem_type']}
- Complexity: {problem_info['complexity']}
- Specification: {problem_info['specification']}

Available Knowledge:
{rag_context}

Previous Failures:
{self._get_failure_context()}

Generate 4 different reasoning approaches for this Lean 4 problem. For each approach, provide:
1. High-level strategy
2. Implementation approach
3. Proof strategy
4. Confidence level (0-1)
5. Rationale for this approach

Focus on:
- Using Lean 4 syntax correctly
- Choosing appropriate tactics
- Handling edge cases
- Making proofs constructive and complete
"""
        
        planning_prompt = f"""
You are an expert Lean 4 theorem prover. Analyze this problem and generate 4 distinct reasoning paths.

{planning_context}

Respond in this JSON format:
{{
  "paths": [
    {{
      "approach": "direct_implementation",
      "strategy": "Direct algorithmic implementation",
      "code_approach": "Use straightforward conditional logic",
      "proof_strategy": "Use simp and cases for proof",
      "confidence": 0.8,
      "rationale": "Simple and reliable approach"
    }},
    {{
      "approach": "mathlib_approach", 
      "strategy": "Leverage Mathlib library functions",
      "code_approach": "Use existing library functions like min, max",
      "proof_strategy": "Use Mathlib lemmas and existing proofs",
      "confidence": 0.9,
      "rationale": "Robust using proven library components"
    }},
    {{
      "approach": "specification_driven",
      "strategy": "Design implementation to make proof easier",
      "code_approach": "Structure code to match specification exactly",
      "proof_strategy": "Unfold definitions and use constructor",
      "confidence": 0.7,
      "rationale": "Proof-oriented design approach"
    }},
    {{
      "approach": "pattern_matching",
      "strategy": "Use pattern matching and recursion",
      "code_approach": "Break problem into cases using pattern matching",
      "proof_strategy": "Use induction and case analysis",
      "confidence": 0.6,
      "rationale": "Functional programming approach"
    }}
  ]
}}
"""
        
        response = self.planning_agent.get_response([{"role": "user", "content": planning_prompt}])
        
        # Parse response and create thought nodes
        thought_nodes = []
        try:
            parsed = self._parse_json_response(response)
            paths = parsed.get("paths", [])
            
            for path_data in paths:
                approach_name = path_data.get("approach", "direct_implementation")
                try:
                    approach = ReasoningPath(approach_name)
                except ValueError:
                    approach = ReasoningPath.DIRECT_IMPLEMENTATION
                
                node = ThoughtNode(
                    path=approach,
                    reasoning=path_data.get("strategy", ""),
                    code_approach=path_data.get("code_approach", ""),
                    proof_strategy=path_data.get("proof_strategy", ""),
                    confidence=float(path_data.get("confidence", 0.5)),
                    rationale=path_data.get("rationale", "")
                )
                thought_nodes.append(node)
        
        except Exception as e:
            print(f"[ToT Agent] Failed to parse planning response: {e}")
            # Create default thought nodes
            thought_nodes = self._create_default_thought_nodes()
        
        # Sort by confidence
        thought_nodes.sort(key=lambda x: x.confidence, reverse=True)
        print(f"[ToT Agent] Generated {len(thought_nodes)} thought nodes")
        
        return thought_nodes
    
    def _create_default_thought_nodes(self) -> List[ThoughtNode]:
        """Create default thought nodes if planning fails"""
        return [
            ThoughtNode(
                path=ReasoningPath.DIRECT_IMPLEMENTATION,
                reasoning="Direct implementation approach",
                code_approach="Use straightforward logic",
                proof_strategy="Use basic tactics like simp and rfl",
                confidence=0.8,
                rationale="Simple and reliable"
            ),
            ThoughtNode(
                path=ReasoningPath.MATHLIB_APPROACH,
                reasoning="Use Mathlib functions",
                code_approach="Leverage existing library functions",
                proof_strategy="Use Mathlib lemmas",
                confidence=0.7,
                rationale="Robust library approach"
            )
        ]
    
    def implement_solution(self, node: ThoughtNode, problem_info: Dict[str, str], rag_context: str) -> Solution:
        """Implement a solution following a specific thought node"""
        print(f"[ToT Agent] Implementing {node.path.value} approach...")
        
        reasoning_trace = [f"Following {node.path.value} approach: {node.reasoning}"]
        
        # Generate code
        code = self._generate_code(node, problem_info, rag_context, reasoning_trace)
        
        # Generate proof  
        proof = self._generate_proof(node, problem_info, code, rag_context, reasoning_trace)
        
        solution = Solution(
            code=code,
            proof=proof,
            approach=node.path.value,
            reasoning_trace=reasoning_trace
        )
        
        return solution
    
    def _generate_code(self, node: ThoughtNode, problem_info: Dict[str, str], rag_context: str, reasoning_trace: List[str]) -> str:
        """Generate implementation code using the thought node strategy"""
        
        code_prompt = f"""
Generate Lean 4 implementation code using the {node.path.value} approach.

Problem Context:
- Function: {problem_info['function_name']} {problem_info['parameters']} : {problem_info['return_type']}
- Problem Type: {problem_info['problem_type']}
- Specification: {problem_info['specification']}

Strategy: {node.reasoning}
Code Approach: {node.code_approach}

Available Knowledge:
{rag_context}

Requirements:
1. Generate ONLY the implementation code (no function signature)
2. Follow the {node.path.value} strategy
3. Handle all edge cases mentioned in the specification
4. Use correct Lean 4 syntax
5. Do NOT use 'sorry' in implementation

For {node.path.value} approach:
{self._get_approach_guidance(node.path)}

Generate the implementation code:
"""
        
        response = self.generation_agent.get_response([{"role": "user", "content": code_prompt}])
        code = self._clean_code_response(response)
        
        reasoning_trace.append(f"Generated code using {node.code_approach}")
        return code
    
    def _generate_proof(self, node: ThoughtNode, problem_info: Dict[str, str], code: str, rag_context: str, reasoning_trace: List[str]) -> str:
        """Generate proof using the thought node strategy"""
        
        proof_prompt = f"""
Generate Lean 4 proof using the {node.path.value} approach.

Problem Context:
- Function: {problem_info['function_name']}
- Implementation: {code}
- Specification: {problem_info['specification']}

Proof Strategy: {node.proof_strategy}

Available Knowledge:
{rag_context}

Requirements:
1. Generate ONLY the proof tactics (no 'by' keyword)
2. Start with 'unfold {problem_info['function_name']} {problem_info['function_name']}_spec'
3. Use the strategy: {node.proof_strategy}
4. Make the proof complete and constructive
5. Do NOT use 'sorry' in proof

For {node.path.value} proof approach:
{self._get_proof_guidance(node.path)}

Generate the proof tactics:
"""
        
        response = self.generation_agent.get_response([{"role": "user", "content": proof_prompt}])
        proof = self._clean_proof_response(response)
        
        reasoning_trace.append(f"Generated proof using {node.proof_strategy}")
        return proof
    
    def _get_approach_guidance(self, path: ReasoningPath) -> str:
        """Get specific guidance for each approach"""
        guidance = {
            ReasoningPath.DIRECT_IMPLEMENTATION: """
- Use if-then-else for conditionals
- Use basic arithmetic operations
- Keep logic simple and clear
- Handle edge cases explicitly""",
            
            ReasoningPath.MATHLIB_APPROACH: """
- Use library functions like min, max, Array.map, Array.any, Array.all
- Leverage existing proven implementations
- Import necessary modules
- Use standard library patterns""",
            
            ReasoningPath.SPECIFICATION_DRIVEN: """
- Design implementation to match specification structure
- Make proof obligations obvious from code structure
- Use types that align with specification
- Structure conditionals to match proof cases""",
            
            ReasoningPath.PATTERN_MATCHING: """
- Use pattern matching on input structure
- Handle base cases and recursive cases
- Use match expressions for complex logic
- Structure recursion carefully"""
        }
        return guidance.get(path, "Use sound programming principles")
    
    def _get_proof_guidance(self, path: ReasoningPath) -> str:
        """Get specific proof guidance for each approach"""
        guidance = {
            ReasoningPath.DIRECT_IMPLEMENTATION: """
- Use simp for simplification
- Use cases for conditionals
- Use constructor for conjunctions
- Use exact for direct proofs""",
            
            ReasoningPath.MATHLIB_APPROACH: """
- Use existing Mathlib lemmas
- Apply library theorems
- Use rw with Mathlib results
- Leverage proven properties""",
            
            ReasoningPath.SPECIFICATION_DRIVEN: """
- Unfold all definitions first
- Use constructor to build proofs
- Match proof structure to specification structure
- Use exact for obvious goals""",
            
            ReasoningPath.PATTERN_MATCHING: """
- Use induction for recursive structures
- Use cases for pattern matches
- Handle base cases first
- Use recursive hypotheses"""
        }
        return guidance.get(path, "Use appropriate proof tactics")
    
    def verify_solution(self, solution: Solution, task_lean_code: str) -> bool:
        """Verify solution by executing Lean code"""
        try:
            test_code = task_lean_code.replace("{{code}}", solution.code)
            test_code = test_code.replace("{{proof}}", solution.proof)
            
            result = execute_lean_code(test_code)
            
            if "executed successfully" in result or "No errors found" in result:
                solution.success = True
                solution.reasoning_trace.append("Verification successful")
                return True
            else:
                error_msg = result.split("Lean Error:")[-1].strip() if "Lean Error:" in result else result
                solution.error = error_msg
                solution.reasoning_trace.append(f"Verification failed: {error_msg[:100]}...")
                self.failed_attempts.append(solution)
                return False
                
        except Exception as e:
            solution.error = str(e)
            solution.reasoning_trace.append(f"Verification error: {str(e)}")
            return False
    
    def solve_with_tree_of_thoughts(self, problem_description: str, task_lean_code: str) -> Solution:
        """Main Tree of Thoughts solving process"""
        print("[ToT Agent] Starting Tree of Thoughts solving process...")
        
        # Extract problem information
        problem_info = self.extract_problem_info(problem_description, task_lean_code)
        print(f"[ToT Agent] Analyzing {problem_info['function_name']} ({problem_info['problem_type']})")
        
        # Get relevant knowledge from RAG
        rag_query = f"{problem_info['function_name']} {problem_info['problem_type']} {problem_info['return_type']}"
        rag_context = self._query_rag(rag_query)
        
        # Generate thought nodes (reasoning paths)
        thought_nodes = self.generate_thought_nodes(problem_info, rag_context)
        
        # Try each approach in order of confidence
        for i, node in enumerate(thought_nodes):
            print(f"[ToT Agent] Trying approach {i+1}/{len(thought_nodes)}: {node.path.value}")
            
            # Implement solution for this thought node
            solution = self.implement_solution(node, problem_info, rag_context)
            
            # Verify the solution
            if self.verify_solution(solution, task_lean_code):
                print(f"[ToT Agent] Success with {node.path.value} approach!")
                return solution
            else:
                print(f"[ToT Agent] {node.path.value} approach failed: {solution.error[:50]}...")
                
                # Try to improve solution based on error
                if i < len(thought_nodes) - 1:  # Not the last attempt
                    improved_solution = self._attempt_error_correction(solution, problem_info, rag_context)
                    if improved_solution and self.verify_solution(improved_solution, task_lean_code):
                        print(f"[ToT Agent] Success with improved {node.path.value} approach!")
                        return improved_solution
        
        # All approaches failed
        print("[ToT Agent] All approaches failed")
        return Solution(
            code="sorry",
            proof="sorry", 
            approach="failed",
            reasoning_trace=["All Tree of Thoughts approaches failed"],
            success=False,
            error="No successful approach found"
        )
    
    def _attempt_error_correction(self, failed_solution: Solution, problem_info: Dict[str, str], rag_context: str) -> Optional[Solution]:
        """Attempt to correct errors based on feedback"""
        print("[ToT Agent] Attempting error correction...")
        
        correction_prompt = f"""
The previous solution failed. Analyze the error and provide a corrected version.

Failed Solution:
Code: {failed_solution.code}
Proof: {failed_solution.proof}
Error: {failed_solution.error}

Problem Context:
- Function: {problem_info['function_name']}
- Specification: {problem_info['specification']}

Provide corrected code and proof. Focus on fixing the specific error mentioned.

Respond in JSON format:
{{
  "corrected_code": "...",
  "corrected_proof": "...",
  "fix_explanation": "..."
}}
"""
        
        try:
            response = self.verification_agent.get_response([{"role": "user", "content": correction_prompt}])
            parsed = self._parse_json_response(response)
            
            if "corrected_code" in parsed and "corrected_proof" in parsed:
                corrected = Solution(
                    code=parsed["corrected_code"],
                    proof=parsed["corrected_proof"],
                    approach=f"{failed_solution.approach}_corrected",
                    reasoning_trace=failed_solution.reasoning_trace + [f"Error correction: {parsed.get('fix_explanation', 'Applied fixes')}"]
                )
                return corrected
        except Exception as e:
            print(f"[ToT Agent] Error correction failed: {e}")
        
        return None
    
    def _get_failure_context(self) -> str:
        """Get context from previous failures"""
        if not self.failed_attempts:
            return "No previous failures"
        
        context = "Previous failure patterns:\n"
        for attempt in self.failed_attempts[-3:]:  # Last 3 failures
            context += f"- {attempt.approach}: {attempt.error[:100]}...\n"
        return context
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {}
        except (json.JSONDecodeError, AttributeError):
            print(f"[ToT Agent] Failed to parse JSON: {response[:100]}...")
            return {}
    
    def _clean_code_response(self, response: str) -> str:
        """Clean and extract code from LLM response"""
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        
        lines = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('def ')]
        
        return '\n'.join(lines) if lines else response.strip()
    
    def _clean_proof_response(self, response: str) -> str:
        """Clean and extract proof from LLM response"""
        response = re.sub(r'```lean\n?', '', response)
        response = re.sub(r'```\n?', '', response)
        response = re.sub(r'^\s*by\s+', '', response.strip())
        
        lines = [line.strip() for line in response.split('\n') 
                if line.strip() and not line.strip().startswith('--')]
        
        return '\n'.join(lines) if lines else response.strip()


def main_workflow(problem_description: str, task_lean_code: str = "") -> Dict[str, str]:
    """
    Tree of Thoughts main workflow for Lean 4 theorem proving.
    
    This implements a genuine multi-agent system with dynamic reasoning,
    replacing any hardcoded solutions with AI-generated approaches.
    """
    print("[Main Workflow] Starting Tree of Thoughts Multi-Agent System...")
    
    # Validate inputs
    if not task_lean_code or not problem_description:
        print("[Main Workflow] Missing required inputs")
        return {"code": "sorry", "proof": "sorry"}
    
    # Initialize Tree of Thoughts agent
    tot_agent = TreeOfThoughtsAgent()
    
    try:
        # Solve using Tree of Thoughts approach
        solution = tot_agent.solve_with_tree_of_thoughts(problem_description, task_lean_code)
        
        # Log the reasoning process
        print(f"[Main Workflow] Final approach: {solution.approach}")
        print(f"[Main Workflow] Success: {solution.success}")
        if solution.reasoning_trace:
            print(f"[Main Workflow] Reasoning: {' → '.join(solution.reasoning_trace[-3:])}")
        
        return {
            "code": solution.code,
            "proof": solution.proof
        }
        
    except Exception as e:
        print(f"[Main Workflow] Unexpected error: {e}")
        return {"code": "sorry", "proof": "sorry"}


# Keep existing utility functions for compatibility
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