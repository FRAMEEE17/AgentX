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

(Above command takes ~2 minutes on a MacBook Air with M2 chip and 8GB RAM)

# 🚀 Quick Start (Skip Lean Verification)
If you want to test the AI logic immediately without waiting for full Lean setup:
```bash
cd lab2-starter-code/
```
# Skip Lean verification during development
```bash
export SKIP_LEAN_VERIFICATION=true

# Run tests (will work without Lean installed)
```bash
make test
Add to your shell profile for persistence:
bash# For bash users
echo 'export SKIP_LEAN_VERIFICATION=true' >> ~/.bashrc
source ~/.bashrc
```

# For zsh users  
```bash
echo 'export SKIP_LEAN_VERIFICATION=true' >> ~/.zshrc
source ~/.zshrc
```