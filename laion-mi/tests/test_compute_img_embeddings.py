from compute_img_embeddings import load_parquets

def test_load_parquets():
    table = load_parquets("data/pokemon-split")
    assert table.shape == (633, 3)
    assert table.column_names == ["caption", "jpg"]
    assert table[0]["caption"].as_py() == "A blue and yellow Pokemon with a white belly."