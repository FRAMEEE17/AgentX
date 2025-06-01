<<<<<<< HEAD
# Instructions to install Lean environment

Required OS: Linux (WSL) or MacOS

## Step 1: Install Elan
```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

## Step 2: Install Lean 4 and dependencies
```bash
elan toolchain install v4.18.0
lake update # This might take > 10 minutes
```

## Step 3: Install Lean 4 VSCode extension

Install extension Lean 4 by searching `leanprover.lean4` in the VSCode extension marketplace.

## Step 4: Verify Lean 4 installation
```bash
lake --version
lake lean Lean4CodeGenerator.lean
```

=======
# Instructions to install Lean environment

Required OS: Linux (WSL) or MacOS

## Step 1: Install Elan
```bash
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
```

## Step 2: Install Lean 4 and dependencies
```bash
elan toolchain install v4.18.0
lake update # This might take > 10 minutes
```

## Step 3: Install Lean 4 VSCode extension

Install extension Lean 4 by searching `leanprover.lean4` in the VSCode extension marketplace.

## Step 4: Verify Lean 4 installation
```bash
lake --version
lake lean Lean4CodeGenerator.lean
```

>>>>>>> 1e9a9961e8fdb46ae9c2557929ff8e564c9c54ed
(Above command takes ~2 minutes on a MacBook Air with M2 chip and 8GB RAM)