"""Every module in the package imports."""

import importlib
import pkgutil

import race_auv_factorgraph


def test_all_modules_import():
    for mod in pkgutil.walk_packages(race_auv_factorgraph.__path__, 'race_auv_factorgraph.'):
        importlib.import_module(mod.name)
