import subprocess
import tempfile
import os
import shutil
from pathlib import Path

def execute_lean_code(lean_code: str) -> str:
    """
    Execute Lean 4 code and return the result.
    
    Args:
        lean_code: The Lean 4 code to execute
        
    Returns:
        String containing the execution result or error message
    """
    
    # Create a temporary directory for the Lean project
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        try:
            # Create lakefile.lean
            lakefile_content = '''import Lake
open Lake DSL

package «temp_project» where
  -- add package configuration options here

lean_lib «TempProject» where
  -- add library configuration options here

@[default_target]
lean_exe «temp_project» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
'''
            
            with open(temp_path / "lakefile.lean", "w") as f:
                f.write(lakefile_content)
            
            # Create the main Lean file
            lean_file_path = temp_path / "Main.lean"
            with open(lean_file_path, "w") as f:
                f.write(lean_code)
            
            # Change to the temporary directory
            original_cwd = os.getcwd()
            os.chdir(temp_path)
            
            try:
                # Update lake dependencies (suppress output for cleaner logs)
                subprocess.run(
                    ["lake", "update"], 
                    capture_output=True, 
                    text=True, 
                    timeout=60,
                    check=False  # Don't raise exception on non-zero exit
                )
                
                # Build and run the Lean code
                result = subprocess.run(
                    ["lake", "build"], 
                    capture_output=True, 
                    text=True, 
                    timeout=120
                )
                
                if result.returncode == 0:
                    # If build succeeds, try to run
                    run_result = subprocess.run(
                        ["lake", "exe", "temp_project"],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if run_result.returncode == 0:
                        return f"Lean code executed successfully.\nOutput: {run_result.stdout}"
                    else:
                        return f"Lean code compiled but failed to run.\nError: {run_result.stderr}"
                else:
                    # Extract meaningful error from stderr
                    error_output = result.stderr.strip()
                    if not error_output:
                        error_output = result.stdout.strip()
                    
                    return f"Lean Error: {error_output}"
                    
            finally:
                # Always return to original directory
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            return "Lean Error: Execution timed out"
        except FileNotFoundError:
            return "Error: Lean executable not found. Please install Lean 4 with 'curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh'"
        except Exception as e:
            return f"Error: {str(e)}"


def test_lean_installation():
    """Test if Lean 4 is properly installed."""
    try:
        # Test lean command
        lean_result = subprocess.run(
            ["lean", "--version"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        # Test lake command  
        lake_result = subprocess.run(
            ["lake", "--version"],
            capture_output=True,
            text=True, 
            timeout=10
        )
        
        if lean_result.returncode == 0 and lake_result.returncode == 0:
            return True, f"Lean {lean_result.stdout.strip()}, Lake {lake_result.stdout.strip()}"
        else:
            return False, "Lean or Lake command failed"
            
    except FileNotFoundError:
        return False, "Lean or Lake not found in PATH"
    except Exception as e:
        return False, f"Error testing Lean installation: {e}"


if __name__ == "__main__":
    # Test the installation
    success, message = test_lean_installation()
    print(f"Lean installation test: {'✅ PASS' if success else '❌ FAIL'}")
    print(f"Details: {message}")
    
    if success:
        # Test with simple Lean code
        test_code = '''
import Mathlib

def test_function (x : Nat) : Nat := x

#check test_function

theorem test_theorem : test_function 5 = 5 := by
  unfold test_function
  rfl

#check test_theorem
'''
        
        print("\n🧪 Testing Lean execution...")
        result = execute_lean_code(test_code)
        print(f"Result: {result}")
    else:
        print("\n❌ Please install Lean 4 first:")
        print("curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh")
        print("source ~/.profile")