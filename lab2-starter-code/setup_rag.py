#!/usr/bin/env python3
"""
Enhanced RAG setup with genuine knowledge patterns (no hardcoded solutions)
"""

import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.embedding_db import VectorDB
    from src.embedding_models import OpenAIEmbeddingModel
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def create_lean4_knowledge_base():
    """Create knowledge base with Lean 4 patterns and techniques (no solutions)"""
    
    # Ensure documents directory exists
    os.makedirs("documents", exist_ok=True)
    
    # Create comprehensive Lean 4 knowledge
    create_lean4_syntax_guide()
    create_proof_strategy_guide()
    create_common_patterns_guide()
    create_error_recovery_guide()
    create_mathlib_reference()
    
    print("✅ Created comprehensive Lean 4 knowledge base")


def create_lean4_syntax_guide():
    """Create Lean 4 syntax and structure guide"""
    content = """# Lean 4 Syntax and Structure Guide

## Function Definitions
Basic function structure in Lean 4:
def functionName (param1 : Type1) (param2 : Type2) : ReturnType := implementation

Examples of function signatures:
- def identity (x : Nat) : Nat := ...
- def compare (a b : Int) : Bool := ...
- def process (arr : Array Int) : Array Int := ...

<EOC>

## Conditional Logic Patterns
If-then-else syntax:
if condition then value1 else value2

Pattern for boolean conditions:
if a ≤ b then ... else ...
if x > 0 then ... else ...

Multiple conditions:
if cond1 then val1 else if cond2 then val2 else val3

<EOC>

## Type System Fundamentals
Common types in Lean 4:
- Nat: Natural numbers (0, 1, 2, ...)
- Int: Integers (..., -1, 0, 1, ...)
- Bool: Boolean values (true, false)
- Array T: Arrays of type T
- Prop: Propositions for specifications

Type conversion:
- ↑n converts Nat to Int
- Int.natAbs converts Int to Nat

<EOC>

## Array Operations
Basic array operations:
- arr.size : Get array length
- arr[i]! : Access element at index i
- arr.map f : Apply function f to all elements
- arr.any p : Check if any element satisfies predicate p
- arr.all p : Check if all elements satisfy predicate p
- arr.contains x : Check if array contains element x

Array creation:
#[elem1, elem2, elem3] creates an array

<EOC>

## Arithmetic and Comparison
Basic arithmetic:
- Addition: a + b
- Subtraction: a - b  
- Multiplication: a * b
- Division: a / b
- Modulo: a % b

Comparison operators:
- Equality: a = b
- Less than: a < b
- Less than or equal: a ≤ b (typed as \le)
- Greater than: a > b
- Greater than or equal: a ≥ b (typed as \ge)

<EOC>

## Logical Operators
Logical operations in Lean 4:
- And: ∧ (typed as \and)
- Or: ∨ (typed as \or)  
- Not: ¬ (typed as \not)
- Implication: → (typed as \to)
- Equivalence: ↔ (typed as \iff)

Boolean operations:
- true, false
- &&, || for boolean and/or
- ! for boolean not

<EOC>"""
    
    with open("documents/lean4_syntax.txt", "w") as f:
        f.write(content)


def create_proof_strategy_guide():
    """Create proof strategy and tactics guide"""
    content = """# Lean 4 Proof Strategies and Tactics

## Basic Proof Structure
Standard proof template:
theorem name (params) : statement := by
  tactics

Example structure:
theorem func_correct (input) : specification := by
  unfold func func_spec
  proof_tactics

<EOC>

## Essential Tactics Overview
Core tactics for theorem proving:
- unfold: Expand function and specification definitions
- simp: Simplify expressions using built-in rules
- rfl: Prove reflexive equality (a = a)
- exact: Provide exact proof term
- constructor: Build conjunction (A ∧ B)
- left/right: Choose disjunction branch (A ∨ B)
- cases: Case analysis on conditions
- by_cases: Split proof by boolean condition

<EOC>

## Proof Strategy for Specifications
General approach to proving specifications:
1. Start with: unfold function_name function_spec
2. Simplify with: simp
3. Handle conditionals with: by_cases h : condition
4. Build conjunctions with: constructor
5. Prove each part separately
6. Use exact or rfl for final steps

<EOC>

## Conditional Logic Proofs
Proving properties of if-then-else:
by_cases h : condition
· simp [h]  -- Case when condition is true
  prove_true_case
· simp [h]  -- Case when condition is false  
  prove_false_case

Common pattern for boolean functions:
simp only [decide_eq_true_iff]

<EOC>

## Array Proof Patterns
Proving array properties:
1. Size preservation: simp [Array.size_map]
2. Element access: simp [Array.getElem_map]
3. Existential properties: use specific indices
4. Universal properties: intro i h; prove for arbitrary i

Array membership proofs:
- Use Array.any_eq_true for existence
- Use Array.all_eq_true for universals
- Use Array.contains_def for membership

<EOC>

## Arithmetic Proof Techniques
Proving arithmetic properties:
- Use linarith for linear arithmetic
- Use norm_num for numerical computations
- Use ring for ring operations
- Use exact Nat.zero_le _ for non-negativity
- Use Nat.mod_lt for modulo bounds

<EOC>

## Error Recovery Strategies
When proofs fail:
1. Check if all definitions are unfolded
2. Verify simp is applied appropriately
3. Ensure all cases are handled
4. Check for missing constructors
5. Add intermediate lemmas if needed

Common fixes:
- Add push_neg at h for negation handling
- Use exact h for direct hypothesis application
- Use assumption when goal matches hypothesis

<EOC>"""

    with open("documents/lean4_proofs.txt", "w") as f:
        f.write(content)


def create_common_patterns_guide():
    """Create guide for common programming patterns"""
    content = """# Common Lean 4 Programming Patterns

## Identity and Basic Functions
Identity function pattern:
- Input: single parameter
- Output: same parameter unchanged
- Implementation: return the parameter directly

Arithmetic function patterns:
- Binary operations: combine two inputs with operator
- Unary operations: transform single input
- Comparison functions: return boolean result

<EOC>

## Comparison and Ordering
Minimum/maximum patterns:
- Two-way comparison: use conditional on ≤ relation
- Multi-way comparison: nested conditionals or library functions
- Array min/max: use folding or library functions

Boolean logic patterns:
- Opposite conditions: check sign combinations
- Divisibility: use modulo operation (n % d = 0)
- Range checking: combine upper and lower bounds

<EOC>

## Array Processing Patterns
Transformation patterns:
- Element-wise transformation: use Array.map
- Filtering: use Array.filter
- Reduction: use Array.foldl or Array.foldr

Search patterns:
- Existence check: use Array.any
- Universal check: use Array.all  
- Membership test: use Array.contains
- Element finding: use Array.find?

<EOC>

## Mathematical Operations
Digit extraction:
- Last digit: use modulo 10 operation
- Digit separation: combine division and modulo
- Number reconstruction: use powers of 10

Geometric calculations:
- Area computations: multiply dimensions
- Volume calculations: multiply length × width × height
- Surface area: sum of face areas

<EOC>

## Conditional Logic Design
Two-branch conditionals:
- Compare values and return appropriate branch
- Handle edge cases (equality, zero, etc.)
- Ensure all paths return correct type

Multi-branch logic:
- Use nested if-then-else for multiple conditions
- Consider all possible cases
- Optimize for readability and proof simplicity

<EOC>

## Type-Driven Development
Design by specification:
- Start with the specification requirements
- Choose implementation that makes proof easier
- Structure code to match proof obligations
- Use types to enforce correctness

Proof-oriented programming:
- Make invariants explicit in code structure
- Choose representations that simplify proofs
- Avoid unnecessary complexity in implementation

<EOC>"""

    with open("documents/lean4_patterns.txt", "w") as f:
        f.write(content)


def create_error_recovery_guide():
    """Create error recovery and debugging guide"""
    content = """# Lean 4 Error Recovery and Debugging

## Common Compilation Errors
Syntax errors:
- Check parentheses and bracket matching
- Verify function signature syntax
- Ensure proper import statements
- Check for typos in keywords (def, theorem, by)

Type errors:
- Verify parameter types match usage
- Check return type consistency
- Use type annotations when needed
- Convert between Nat and Int as needed

<EOC>

## Proof Errors and Solutions
Unsolved goals:
- Check if all cases are handled
- Verify constructor is used for conjunctions
- Ensure left/right for disjunctions
- Add missing exact or assumption

Tactic failures:
- Try different simplification approaches
- Break complex proofs into steps
- Use intermediate lemmas
- Check hypothesis names and types

<EOC>

## Debugging Strategies
Systematic debugging:
1. Isolate the failing component
2. Test with simpler inputs
3. Check each proof step individually
4. Use sorry temporarily to isolate issues
5. Verify all imports are correct

Proof debugging:
- Add intermediate proof steps
- Use #check to verify types
- Test with concrete examples
- Simplify complex expressions

<EOC>

## Recovery from Failed Approaches
When direct approach fails:
- Try using library functions instead
- Simplify the implementation
- Break problem into smaller parts
- Use different proof strategies
- Consider alternative data representations

Fallback strategies:
- Use more basic tactics instead of automation
- Avoid complex metaprogramming
- Use classical logic if constructive fails
- Simplify specifications if possible

<EOC>

## Best Practices for Robust Code
Defensive programming:
- Handle edge cases explicitly
- Use clear, readable implementations
- Avoid overly clever optimizations
- Choose simple over complex when possible

Maintainable proofs:
- Use descriptive variable names
- Break long proofs into lemmas
- Comment complex proof steps
- Structure proofs logically

<EOC>"""

    with open("documents/lean4_debugging.txt", "w") as f:
        f.write(content)


def create_mathlib_reference():
    """Create Mathlib library reference"""
    content = """# Lean 4 Mathlib Library Reference

## Essential Library Functions
Comparison and ordering:
- min: minimum of two values
- max: maximum of two values  
- Nat.min, Int.min: type-specific minimums
- Nat.max, Int.max: type-specific maximums

Arithmetic operations:
- Standard operators: +, -, *, /, %
- Nat.mod: natural number modulo
- Int.natAbs: absolute value to natural number
- Nat.pow: exponentiation

<EOC>

## Array Library Functions
Core array operations:
- Array.size: get array length
- Array.get!: safe element access
- Array.map: transform all elements
- Array.filter: select elements by predicate

Search and test operations:
- Array.any: test if any element satisfies condition
- Array.all: test if all elements satisfy condition
- Array.contains: membership testing
- Array.find?: find first matching element

<EOC>

## Useful Lemmas for Proofs
Comparison lemmas:
- min_le_left: min a b ≤ a
- min_le_right: min a b ≤ b
- le_min_iff: c ≤ min a b ↔ c ≤ a ∧ c ≤ b
- min_comm: min a b = min b a

Array lemmas:
- Array.size_map: (arr.map f).size = arr.size
- Array.getElem_map: (arr.map f)[i] = f (arr[i])
- Array.any_eq_true: equivalence for existence
- Array.all_eq_true: equivalence for universals

<EOC>

## Natural Number Properties
Basic properties:
- Nat.zero_le: 0 ≤ n for any natural n
- Nat.mod_lt: n % m < m when m > 0
- Nat.div_mul_cancel: cancellation properties
- Nat.mod_add_div: division algorithm

Useful for digit operations:
- Properties of modulo 10
- Division by powers of 10
- Digit extraction techniques

<EOC>

## Boolean Decision Procedures
Decidability:
- decide_eq_true_iff: converts decidable propositions
- Bool.decide: proposition to boolean conversion
- Classical reasoning when needed

Common decidable propositions:
- Equality on basic types
- Ordering relations
- Arithmetic properties
- Boolean combinations

<EOC>

## Proof Automation
Simplification tactics:
- simp: uses simp lemmas automatically
- simp only [lemmas]: restricted simplification
- norm_num: numerical computations
- ring: ring equation solving

Linear arithmetic:
- linarith: linear arithmetic decision procedure
- omega: linear arithmetic over integers
- Use for comparison and arithmetic goals

<EOC>"""

    with open("documents/mathlib_reference.txt", "w") as f:
        f.write(content)


def main():
    """Main setup function"""
    print("=" * 60)
    print("Enhanced RAG Setup for Tree of Thoughts Lab 2")
    print("=" * 60)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("   Please set your OpenAI API key to use the RAG system:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Create knowledge base
    create_lean4_knowledge_base()
    
    # Initialize RAG database
    print("Creating embeddings database...")
    try:
        embedding_model = OpenAIEmbeddingModel()
        
        vector_db = VectorDB(
            directory="documents",
            vector_file="embeddings.npy",
            embedding_model=embedding_model
        )
        
        print("✅ Enhanced RAG database created successfully!")
        print("✅ Knowledge base contains Lean 4 techniques and patterns")
        print("✅ No hardcoded solutions - only genuine knowledge")
        return True
        
    except Exception as e:
        print(f"❌ Error creating RAG database: {e}")
        return False


if __name__ == "__main__":
    success = main()
    print("\n📋 Next Steps:")
    print("   1. Replace src/main.py with the Tree of Thoughts implementation")
    print("   2. Test with: python test_runner.py --task task_id_0")
    print("   3. The system now uses genuine AI reasoning instead of hardcoding")
    sys.exit(0 if success else 1)