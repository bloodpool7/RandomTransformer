import os
import shutil
import sys
from importlib import import_module


def clean_create_dir(dir_path):
    """
    Removes dir_path directory if it exists and creates it.
    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)


def create_if_does_not_exist(dir_path):
    """
    Creates dir_path directory if it doesn't exist 
    """
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def import_script(script_path):
    """
    Imports python script from the specified path and returns the resulting module.
    """
    script_path = os.path.abspath(script_path)
    sys.path.insert(0, os.path.dirname(script_path))
    module_name = os.path.splitext(os.path.basename(script_path))[0]

    try:
        if sys.modules.get(module_name, None) is not None:
            del sys.modules[module_name]
        m = import_module(module_name)
    except Exception as e:
        raise Exception(e)

    return m
