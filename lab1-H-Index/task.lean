def hIndex (citations : Array Nat) : Nat :=
  -- << CODE START >>
  let n := citations.size
  let rec findHIndex (h : Nat) : Nat :=
    if h = 0 then
      0
    else
      let count := citations.foldl (fun acc c => if c >= h then acc + 1 else acc) 0
      if count >= h then
        h
      else
        findHIndex (h - 1)
  termination_by h

  findHIndex n
  -- << CODE END >>

def hIndex_spec (citations : Array Nat) (result : Nat) : Prop :=
  -- << SPEC START >>
  let n := citations.size
  let countAtLeast (h : Nat) : Nat :=
    citations.foldl (fun count c => if c >= h then count + 1 else count) 0

  (countAtLeast result >= result) ∧
  (∀ h, h > result → countAtLeast h < h) ∧
  (result <= n)
  -- << SPEC END >>

-- ===================== DEBUG SECTION =====================

-- -- Helper function to count papers with at least h citations
-- def countPapersWithAtLeast (citations : Array Nat) (h : Nat) : Nat :=
--   citations.foldl (fun count c => if c >= h then count + 1 else count) 0

-- -- Debug function to show step-by-step calculation
-- def debugHIndex (citations : Array Nat) : String :=
--   let n := citations.size
--   let steps := List.range (n + 1) |>.reverse |>.map fun h =>
--     let count := countPapersWithAtLeast citations h
--     let valid := if count >= h then "✓" else "✗"
--     s!"h={h}: count={count} {valid}"
--   String.intercalate "\n" steps

-- ===================== TEST CASES =====================

#guard hIndex #[0, 1, 3, 5, 6] = 3
#guard hIndex #[1, 2, 100] = 2
#guard hIndex #[0, 0] = 0
#guard hIndex #[100] = 1

#guard hIndex #[0, 1, 4, 5, 6, 6] = 4
#guard hIndex #[1, 1, 1, 1, 1] = 1
#guard hIndex #[0, 0, 0, 0] = 0
#guard hIndex #[10, 10, 10, 10, 10] = 5
#guard hIndex #[0, 0, 4, 4] = 2
#guard hIndex #[11, 15] = 2
#guard hIndex #[] = 0  -- Empty array

-- -- ===================== EVALUATION TESTS =====================

-- -- Test case 1: [0, 1, 3, 5, 6] should return 3
-- #eval hIndex #[0, 1, 3, 5, 6]
-- #eval countPapersWithAtLeast #[0, 1, 3, 5, 6] 3  -- Should be >= 3
-- #eval countPapersWithAtLeast #[0, 1, 3, 5, 6] 4  -- Should be < 4

-- -- Test case 2: [1, 2, 100] should return 2
-- #eval hIndex #[1, 2, 100]
-- #eval countPapersWithAtLeast #[1, 2, 100] 2  -- Should be >= 2
-- #eval countPapersWithAtLeast #[1, 2, 100] 3  -- Should be < 3

-- -- Test case 3: Edge cases
-- #eval hIndex #[0, 0]
-- #eval hIndex #[100]
-- #eval hIndex #[]

-- -- ===================== STEP-BY-STEP DEBUG =====================

-- -- Show detailed calculation for test case 1
-- #eval s!"Debug for [0, 1, 3, 5, 6]:\n{debugHIndex #[0, 1, 3, 5, 6]}"

-- -- Show detailed calculation for test case 2
-- #eval s!"Debug for [1, 2, 100]:\n{debugHIndex #[1, 2, 100]}"

-- -- Show detailed calculation for edge case
-- #eval s!"Debug for [0, 0]:\n{debugHIndex #[0, 0]}"

-- -- ===================== VERIFICATION TESTS =====================

-- -- Manual verification of h-index definition
-- example : hIndex #[0, 1, 3, 5, 6] = 3 := by rfl
-- example : hIndex #[1, 2, 100] = 2 := by rfl
-- example : hIndex #[0, 0] = 0 := by rfl
-- example : hIndex #[100] = 1 := by rfl

-- -- Verify counting function works correctly
-- example : countPapersWithAtLeast #[0, 1, 3, 5, 6] 3 = 3 := by rfl
-- example : countPapersWithAtLeast #[0, 1, 3, 5, 6] 4 = 2 := by rfl
-- example : countPapersWithAtLeast #[1, 2, 100] 2 = 2 := by rfl
-- example : countPapersWithAtLeast #[1, 2, 100] 3 = 1 := by rfl

-- -- ===================== ADDITIONAL EDGE CASES =====================

-- -- Test with larger arrays
-- #guard hIndex #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] = 5
-- #guard hIndex #[1, 1, 1, 1, 1, 1, 1, 1, 1, 1] = 1
-- #guard hIndex #[10, 9, 8, 7, 6, 5, 4, 3, 2, 1] = 5

-- -- Test boundary conditions
-- #guard hIndex #[0] = 0
-- #guard hIndex #[1] = 1
-- #guard hIndex #[1000] = 1

-- -- Test identical high values
-- #guard hIndex #[100, 100, 100] = 3

-- -- ===================== PERFORMANCE CHECK =====================

-- -- Test with moderately large input
-- def largeTest : Array Nat := Array.range 50 |>.map (fun i => i)
-- #eval hIndex largeTest

-- -- Verify the result makes sense
-- #eval countPapersWithAtLeast largeTest 25
