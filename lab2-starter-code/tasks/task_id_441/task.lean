<<<<<<< HEAD
import Mathlib
import Aesop

-- Implementation
def cubeSurfaceArea (size : Int) : Int :=
  -- << CODE START >>
  {{code}}
  -- << CODE END >>


-- Theorem: The surface area of the cube is calculated correctly
def cubeSurfaceArea_spec (size : Int) (result : Int) : Prop :=
  -- << SPEC START >>
  result = 6 * size * size
  -- << SPEC END >>

theorem cubeSurfaceArea_spec_satisfied (size : Int):
  cubeSurfaceArea_spec size (cubeSurfaceArea size) := by
  -- << PROOF START >>
  unfold cubeSurfaceArea cubeSurfaceArea_spec
  {{proof}}
  -- << PROOF END >>
=======
import Mathlib
import Aesop

-- Implementation
def cubeSurfaceArea (size : Int) : Int :=
  -- << CODE START >>
  {{code}}
  -- << CODE END >>


-- Theorem: The surface area of the cube is calculated correctly
def cubeSurfaceArea_spec (size : Int) (result : Int) : Prop :=
  -- << SPEC START >>
  result = 6 * size * size
  -- << SPEC END >>

theorem cubeSurfaceArea_spec_satisfied (size : Int):
  cubeSurfaceArea_spec size (cubeSurfaceArea size) := by
  -- << PROOF START >>
  unfold cubeSurfaceArea cubeSurfaceArea_spec
  {{proof}}
  -- << PROOF END >>
>>>>>>> 1e9a9961e8fdb46ae9c2557929ff8e564c9c54ed
