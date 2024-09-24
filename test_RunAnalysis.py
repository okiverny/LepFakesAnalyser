def test_treename():
    from LFAConfig import Configurate
    cfg = Configurate("lephad")
    if not isinstance(cfg[0], str):
        raise TypeError("tree_name should be of a string type")
    assert cfg[0] == 'tHqLoop_nominal_Loose', "tree_name should be 'tHqLoop_nominal_Loose'"
