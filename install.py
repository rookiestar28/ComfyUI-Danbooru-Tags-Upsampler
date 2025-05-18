import subprocess
import sys
import os
import logging

# Configure logger for this script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def is_package_installed(package_name):
    """Checks if a package is installed."""
    try:
        # Attempt to import the package. If it fails, it's not installed.
        # For packages with hyphens or different import names, this might need adjustment.
        # A more robust way is to use pip show.
        result = subprocess.run([sys.executable, "-m", "pip", "show", package_name], capture_output=True, text=True, check=False)
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Error checking if {package_name} is installed: {e}")
        return False

def install_package(package_name_with_specifiers, friendly_name=""):
    """Installs a package using pip."""
    if not friendly_name:
        friendly_name = package_name_with_specifiers
    try:
        logger.info(f"Attempting to install {friendly_name}...")
        # Using --no-cache-dir can sometimes help with certain pip install issues
        # Using --upgrade to ensure the version from requirements is met or updated
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", package_name_with_specifiers])
        logger.info(f"{friendly_name} installed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {friendly_name}. Error: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while installing {friendly_name}: {e}")

def install_requirements():
    """
    Installs packages from requirements.txt located in the same directory as this script.
    More robust than installing one by one for complex dependencies.
    """
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_file):
        logger.info(f"Found requirements.txt. Installing all dependencies from it...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "-r", requirements_file])
            logger.info("All dependencies from requirements.txt processed successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies from {requirements_file}. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while installing from {requirements_file}: {e}")
    else:
        logger.warning(f"{requirements_file} not found. Attempting to install known core dependencies individually.")
        # Fallback to individual installs if requirements.txt is missing (though it shouldn't be)
        # This section can be simplified if requirements.txt is always present and preferred.
        core_dependencies = {
            "torch": "torch (PyTorch)", # PyTorch is a base requirement for ComfyUI anyway
            "transformers": "transformers (Hugging Face Transformers)",
            "optimum[onnxruntime]": "optimum with ONNX Runtime support"
            # Add other critical individual fallbacks if necessary
        }
        for pkg_spec, name in core_dependencies.items():
             # A simple check; pip install -r requirements.txt handles existing packages better.
             # This individual check is very basic.
            pkg_name_to_check = pkg_spec.split('[')[0].split('=')[0].split('>')[0].split('<')[0] # Basic name extraction
            if not is_package_installed(pkg_name_to_check): # Simple check, might not cover all cases
                install_package(pkg_spec, name)
            else:
                logger.info(f"{name} (or base '{pkg_name_to_check}') appears to be installed. Skipping individual install, relying on requirements.txt or existing environment.")


if __name__ == "__main__":
    logger.info("Running install script for Danbooru Tags Upsampler (Ray)...")
    # The primary method should be to install from requirements.txt
    install_requirements()
    logger.info("Install script finished.")