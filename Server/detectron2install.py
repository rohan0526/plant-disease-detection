import distutils.core
import os
import sys
from subprocess import call

def install_detectron2():
    # Clone the Detectron2 repository
    call(['git', 'clone', 'https://github.com/facebookresearch/detectron2'])
    
    # Install Detectron2 using its setup.py
    dist = distutils.core.run_setup("./detectron2/setup.py")
    call(['pip', 'install'] + dist.install_requires)
    
    # Add Detectron2 to the Python path
    sys.path.insert(0, os.path.abspath('./detectron2'))

if __name__ == "__main__":
    install_detectron2()
