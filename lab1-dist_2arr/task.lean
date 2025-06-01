def findDistanceValue (arr1 : Array Int) (arr2 : Array Int) (d : Nat) : Nat :=
  -- << CODE START >>
  arr1.foldl (fun count x =>
    let isFarFromAll := arr2.all (fun y => Int.natAbs (x - y) > d)
    if isFarFromAll then count + 1 else count) 0
  -- << CODE END >>

def findDistanceValue_spec (arr1 : Array Int) (arr2 : Array Int) (d : Nat) (result : Nat) : Prop :=
  -- << SPEC START >>
  let countValid := arr1.foldl (fun acc x =>
    if (∀ j, j < arr2.size → Int.natAbs (x - arr2[j]!) > d) then acc + 1 else acc) 0

  result = countValid ∧ result ≤ arr1.size
  -- << SPEC END >>

-- You can use the following to do unit tests, you don't need to submit the following code

#guard findDistanceValue #[4, 5, 8] #[10, 9, 1, 8] 2 = 2
#guard findDistanceValue #[1, 4, 2, 3] #[-4, -3, 6, 10, 20, 30] 3 = 2
#guard findDistanceValue #[2, 1, 100, 3] #[-5, -2, 10, -3, 7] 6 = 1
#guard findDistanceValue #[1, 2, 3] #[4, 5, 6] 1 = 2
#guard findDistanceValue #[1, 2, 3] #[10, 20, 30] 1 = 3
#guard findDistanceValue #[0] #[0] 0 = 0
#guard findDistanceValue #[0] #[1] 0 = 1
#guard findDistanceValue #[-1, -2, -3] #[1, 2, 3] 1 = 3
#guard findDistanceValue #[5, 5, 5] #[5] 0 = 0
#guard findDistanceValue #[1, 2, 3, 4, 5] #[10] 4 = 5
