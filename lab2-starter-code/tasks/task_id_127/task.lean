<<<<<<< HEAD
import Mathlib
import Aesop

-- Implementation
def multiply (a : Int) (b : Int) : Int :=
  -- << CODE START >>
  {{code}}
  -- << CODE END >>


-- Theorem: The result should be the product of the two input integers
def multiply_spec (a : Int) (b : Int) (result : Int) : Prop :=
  -- << SPEC START >>
  result = a * b
  -- << SPEC END >>

theorem multiply_spec_satisfied (a : Int) (b : Int) :
  multiply_spec a b (multiply a b) := by
  -- << PROOF START >>
  unfold multiply multiply_spec
  {{proof}}
  -- << PROOF END >>
=======
import Mathlib
import Aesop

-- Implementation
def multiply (a : Int) (b : Int) : Int :=
  -- << CODE START >>
  {{code}}
  -- << CODE END >>


-- Theorem: The result should be the product of the two input integers
def multiply_spec (a : Int) (b : Int) (result : Int) : Prop :=
  -- << SPEC START >>
  result = a * b
  -- << SPEC END >>

theorem multiply_spec_satisfied (a : Int) (b : Int) :
  multiply_spec a b (multiply a b) := by
  -- << PROOF START >>
  unfold multiply multiply_spec
  {{proof}}
  -- << PROOF END >>
>>>>>>> 1e9a9961e8fdb46ae9c2557929ff8e564c9c54ed
