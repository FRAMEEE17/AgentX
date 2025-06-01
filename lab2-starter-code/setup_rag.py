#!/usr/bin/env python3
"""
Setup script for Lab 2 Lean 4 RAG system.
Run this script to initialize the knowledge base before running the main workflow.
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
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def create_lean4_knowledge_base():    
    print("Setting up Lean 4 RAG Knowledge Base...")
    
    # Ensure documents directory exists
    os.makedirs("documents", exist_ok=True)
    
    # Create Lean 4 specific documentation
    print("Creating Lean 4 documentation files...")
    create_lean4_basics_doc()
    create_lean4_proof_patterns_doc() 
    create_lean4_common_tactics_doc()
    create_lean4_mathlib_examples_doc()
    create_lean4_error_recovery_doc()
    
    # Initialize RAG database
    print("..Creating embeddings database...")
    try:
        embedding_model = OpenAIEmbeddingModel()
        
        vector_db = VectorDB(
            directory="documents",
            vector_file="embeddings.npy",  # Use standard filename
            embedding_model=embedding_model
        )
        
        print("✅ Lean 4 RAG database created successfully!")
        print(f"Database file: embeddings.npy")
        print(f"Chunks file: embeddings_chunks.pkl")
        return vector_db
        
    except Exception as e:
        print(f"❌ Error creating RAG database: {e}")
        print("Make sure your OPENAI_API_KEY is set correctly")
        return None


def create_lean4_basics_doc():
    """Create basic Lean 4 syntax and patterns documentation."""
    content = """# Lean 4 Basic Patterns

## Function Definitions
Simple identity function:
def ident (x : Nat) : Nat := x

Conditional function:
def myMin (a b : Int) : Int := if a ≤ b then a else b

Pattern matching:
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

<EOC>

## Common Data Types
Natural numbers: Nat
Integers: Int  
Booleans: Bool
Arrays: Array T
Lists: List T

Type conversions:
Int.natAbs : Int → Nat
↑n : Nat → Int (coercion)

<EOC>

## Basic Operators
Arithmetic: +, -, *, /, %
Comparison: =, ≠, <, ≤, >, ≥
Logical: ∧, ∨, ¬, →, ↔
Array access: arr[i]!, arr.get! i

<EOC>

## Function Signatures for Common Tasks
Identity: def ident (x : T) : T := x
Minimum of two: def min2 (a b : Int) : Int := if a ≤ b then a else b  
Minimum of three: def min3 (a b c : Int) : Int := min2 (min2 a b) c
Multiplication: def mult (a b : Int) : Int := a * b
Divisibility check: def isDivisible (n d : Int) : Bool := n % d = 0
Array map: def arrayMap (f : T → U) (arr : Array T) : Array U := arr.map f
Common element check: def hasCommon (a b : Array T) : Bool := a.any (fun x => b.contains x)

<EOC>

## Specifications
Simple equality spec:
def func_spec (input : T) (result : U) : Prop := result = expected

Complex spec with multiple conditions:
def complex_spec (input : T) (result : U) : Prop := 
  condition1 ∧ condition2 ∧ result = expected

Array spec pattern:
def array_spec (input : Array T) (result : Array U) : Prop :=
  result.size = input.size ∧ 
  ∀ i, i < input.size → result[i]! = transform input[i]!
"""
    
    with open("documents/lean4_basics.txt", "w") as f:
        f.write(content)


def create_lean4_proof_patterns_doc():
    """Create Lean 4 proof patterns and tactics documentation."""
    content = """# Lean 4 Proof Patterns

## Basic Proof Structure
theorem name (params) : statement := by
  tactics

example (params) : statement := by
  tactics

<EOC>

## Essential Tactics
unfold: Expand definition
simp: Simplify expressions
rw [lemma]: Rewrite using lemma
cases: Case analysis
constructor: Build structure
exact: Provide exact proof
apply: Apply function/theorem
assumption: Use hypothesis
rfl: Reflexivity

<EOC>

## Proving Equality
For definitional equality:
def f (x : Nat) : Nat := x
theorem f_eq (x : Nat) : f x = x := by
  unfold f
  rfl

<EOC>

## Proving Specifications
Basic specification proof pattern:
def func_spec (input : T) (result : U) : Prop := 
  result = expected_value

theorem func_spec_satisfied (input : T) :
  func_spec input (func input) := by
  unfold func func_spec
  rfl

<EOC>

## Conditional Logic Proofs
For if-then-else:
theorem cond_proof (condition : Prop) [Decidable condition] (a b : T) :
  spec (if condition then a else b) := by
  by_cases h : condition
  · simp [h]
    -- prove for true case
  · simp [h]  
    -- prove for false case

<EOC>

## Array Specifications
Array size preservation:
theorem array_size_preserved (arr : Array T) :
  (transform arr).size = arr.size := by
  unfold transform
  simp [Array.size_map]

Element-wise properties:
theorem array_elements (arr : Array T) (i : Nat) (h : i < arr.size) :
  (transform arr)[i]! = transform_element arr[i]! := by
  unfold transform
  simp [Array.getElem_map]

<EOC>

## Logical Connectives
And (∧):
constructor
· -- prove first part
· -- prove second part

Or (∨):
left  -- prove left disjunct
-- or
right -- prove right disjunct

Iff (↔):
constructor
· -- prove forward direction
· -- prove backward direction

<EOC>

## Common Proof Patterns by Problem Type
Identity function:
unfold ident ident_spec
rfl

Arithmetic operations:
unfold operation operation_spec  
ring -- for arithmetic simplification

Comparison operations:
unfold comparison comparison_spec
by_cases h : condition
· simp [h]
· simp [h]

Array operations:
unfold operation operation_spec
simp [Array.size_map, Array.getElem_map]
"""
    
    with open("documents/lean4_proof_patterns.txt", "w") as f:
        f.write(content)


def create_lean4_common_tactics_doc():
    """Create documentation for common Lean 4 tactics and their usage."""
    content = """# Lean 4 Common Tactics Guide

## Simplification Tactics
simp: Automatic simplification using simp lemmas
simp [lemma1, lemma2]: Simplification with specific lemmas
simp only [lemma]: Simplification with only specified lemmas
dsimp: Definitional simplification
norm_num: Normalize numerical expressions

<EOC>

## Rewriting Tactics  
rw [lemma]: Rewrite using equality lemma
rw [← lemma]: Rewrite in reverse direction
rw [lemma] at h: Rewrite in hypothesis h
rw [lemma1, lemma2]: Chain multiple rewrites

<EOC>

## Case Analysis
cases h: Case split on hypothesis h
by_cases h : P: Case split on decidable proposition P
split: Split on if-then-else or match expressions
induction h: Proof by induction

<EOC>

## Goal Management
constructor: Apply constructor of inductive type
exact proof: Provide exact proof term
apply lemma: Apply function/lemma to goal
have h : P := proof: Introduce local hypothesis
suffices h : P: Prove goal by proving P

<EOC>

## Logical Tactics
left/right: Choose disjunct in A ∨ B
exfalso: Prove goal from contradiction
contradiction: Find contradiction in hypotheses
tauto: Solve tautologies
decide: Solve decidable propositions

<EOC>

## Arithmetic Tactics
ring: Solve ring equations
field_simp: Simplify field expressions  
norm_num: Normalize numbers
omega: Linear arithmetic over integers
linarith: Linear arithmetic

<EOC>

## Array-Specific Tactics
Array.ext: Array extensionality
Array.get_set_eq: Array update lemmas
Array.size_*: Array size lemmas
Array.mem_*: Array membership lemmas

<EOC>

## Common Tactic Combinations
For basic equality proofs:
unfold definitions
rfl

For conditional proofs:
by_cases h : condition
· simp [h]
  -- specific proof
· simp [h]
  -- specific proof

For array proofs:
constructor
· -- prove size equality
  simp [Array.size_map]
· -- prove element equality  
  intro i h
  simp [Array.getElem_map]

For specification proofs:
unfold function_name spec_name
-- then appropriate tactics based on definition
"""
    
    with open("documents/lean4_tactics.txt", "w") as f:
        f.write(content)


def create_lean4_mathlib_examples_doc():
    """Create Mathlib usage examples for common operations."""
    content = """# Lean 4 Mathlib Examples

## Integer Operations
min function: min a b (from Mathlib)
max function: max a b (from Mathlib)
absolute value: Int.natAbs n
divisibility: n ∣ m (n divides m)

Useful lemmas:
min_le_left: min a b ≤ a
min_le_right: min a b ≤ b  
le_min_iff: c ≤ min a b ↔ c ≤ a ∧ c ≤ b

<EOC>

## Natural Number Operations
Nat.mod: n % m
Nat.div: n / m
Nat.gcd: greatest common divisor
Nat.lcm: least common multiple

Useful lemmas:
Nat.mod_lt: n % m < m (when m > 0)
Nat.div_mul_cancel: (n * m) / m = n (when m > 0)

<EOC>

## Array Operations from Mathlib
Array.map: Transform elements
Array.filter: Filter elements  
Array.foldl: Left fold
Array.foldr: Right fold
Array.all: Check all elements
Array.any: Check any element
Array.contains: Membership test

<EOC>

## List Operations (Alternative to Arrays)
List.map: Transform elements
List.filter: Filter elements
List.all: Check all elements  
List.any: Check any element
List.mem: Membership test
List.length: Get length

<EOC>

## Common Mathlib Tactics
exact?: Search for exact proof
apply?: Search for applicable lemma
simp?: Show what simp can prove
library_search: Search library for proof
suggest: Suggest applicable tactics

<EOC>

## Boolean Operations
Bool.decide: Convert decidable proposition to Bool
if condition then true else false patterns
Bool.and, Bool.or, Bool.not operations

<EOC>

## Comparison Operations
Decidable equality: a = b (automatically decidable for most types)
Linear order: a < b, a ≤ b, a > b, a ≥ b
min and max operations with associated lemmas

<EOC>

## Working with Decidable Propositions
if h : P then ... else ... (pattern matching on decidability)
Decidable.decide: Convert to boolean
Classical reasoning when needed: classical or by_contra
"""
    
    with open("documents/lean4_mathlib.txt", "w") as f:
        f.write(content)


def create_lean4_error_recovery_doc():
    """Create documentation for common errors and how to fix them."""
    content = """# Lean 4 Error Recovery Guide

## Type Mismatch Errors
Error: "type mismatch"
Solution: Check expected vs actual types
- Use type annotations: (expr : Type)
- Use coercions: ↑n for Nat to Int
- Check function signatures match usage

<EOC>

## Undefined Function Errors  
Error: "unknown identifier"
Solution: Check function is in scope
- Import required modules: import Mathlib
- Check spelling and capitalization
- Ensure function is defined before use

<EOC>

## Proof Errors
Error: "tactic failed" or "unsolved goals"
Solution: Check proof completeness
- Use sorry temporarily to isolate issues
- Check all cases are covered
- Verify hypothesis names and types

<EOC>

## Array Index Errors
Error: "array index out of bounds"
Solution: Provide bounds proofs
- Use h : i < arr.size hypotheses
- Use Array.get! for panic on bounds error
- Use Array.get? for Option return

<EOC>

## Simplification Failures
Error: "simp failed to simplify"
Solution: Add explicit lemmas or use different tactics
- simp [specific_lemma]
- unfold definitions manually
- Use rw instead of simp
- Break into smaller steps

<EOC>

## Pattern Matching Errors
Error: "non-exhaustive pattern"
Solution: Cover all cases
- Add wildcard pattern: | _ => default_case
- Check all constructors are handled
- Use cases tactic in proofs

<EOC>

## Common Quick Fixes
Compilation errors:
1. Check imports at top of file
2. Verify syntax (def, theorem, example keywords)
3. Check parentheses and bracket matching
4. Verify function signatures match implementation

Proof errors:
1. Start with unfold for all custom definitions
2. Use simp for basic arithmetic
3. Use cases/by_cases for conditionals
4. Use constructor for And/Exists goals
5. Use exact for providing terms

<EOC>

## Debugging Strategies
Use #check to verify types
Use #eval to test functions  
Use sorry to isolate proof steps
Use trace tactics to see intermediate goals
Break complex proofs into lemmas

<EOC>

## Recovery from Failed Attempts
If direct approach fails:
1. Try library functions (min, max from Mathlib)
2. Simplify implementation (remove optimizations)
3. Use more basic tactics (avoid advanced automation)
4. Break specification into smaller parts
5. Use classical logic if constructive fails

If proof fails:
1. Check if definition unfolds correctly
2. Try manual case analysis instead of automation
3. Use intermediate lemmas
4. Simplify specification if possible
5. Check for missing imports or lemmas
"""
    
    with open("documents/lean4_errors.txt", "w") as f:
        f.write(content)


def main():
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY environment variable not set!")
        print("   Please set your OpenAI API key to use the RAG system:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        return False
    
    # Create knowledge base
    vector_db = create_lean4_knowledge_base()
    
    if vector_db:
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("   1. Run 'make test' to test your implementation")
        print("   2. Check the generated embeddings.npy and embeddings_chunks.pkl files")
        print("   3. Your main_workflow function is ready to use the RAG system")
        return True
    else:
        print("\n❌ Setup failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)