"""Every module in the package imports."""

import importlib
import pkgutil

import dock_fuser


def test_all_modules_import():
    for mod in pkgutil.walk_packages(dock_fuser.__path__, 'dock_fuser.'):
        importlib.import_module(mod.name)
