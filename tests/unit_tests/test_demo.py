from pathlib import Path
from src.pyfemsolver.solverlib.element import H1Fel


class TestMaterial:
    def setup_method(self):
        print("Setup")

    def teardown_method(self):
        print("Teardown")

    def test_constructor(self):
        fel = H1Fel(order=3)
        assert fel.p == 3
