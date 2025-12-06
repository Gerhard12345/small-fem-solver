from src.pyfemsolver.solverlib.element import H1Fel

class TestElement:
    def setup_method(self):
        print("Setup")

    def teardown_method(self):
        print("Teardown")

    def test_constructor(self):
        p = 3
        fel = H1Fel(order=p)
        assert fel.p == p
        assert fel.ndof_vertex == 3
        assert fel.ndof_faces == 3 * (p - 1)
        assert fel.ndof_facet == p - 1
        assert fel.ndof_inner == int((p - 1) * (p - 2) / 2)
        assert fel.ndof == 3 * p + int((p - 2) * (p - 1) / 2)

    def test_flip_edge(self):
        p = 2
        fel = H1Fel(order=p)
        for edge_nr in (0, 1, 2):
            original_edge = fel.edges[edge_nr]
            fel.flip_edge(edge_nr)
            flipped_edge = fel.edges[edge_nr]
            assert flipped_edge == tuple(reversed(original_edge))