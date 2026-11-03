from copilot.citation_sample import citation_pairs, sample


def test_each_cited_number_in_a_sentence_is_one_pair():
    text = "Atrapos caches overlaps [1]. It beats HRank [2, 3]! Uncited filler."
    assert citation_pairs(text) == [
        ("Atrapos caches overlaps [1].", 1),
        ("It beats HRank [2, 3]!", 2),
        ("It beats HRank [2, 3]!", 3),
    ]


def test_a_number_repeated_in_one_sentence_is_one_pair():
    assert citation_pairs("A [1] and B [1].") == [("A [1] and B [1].", 1)]


def test_an_answer_without_citations_has_no_pairs():
    assert citation_pairs("No brackets here.") == []


def test_the_sample_is_seeded_and_capped():
    pairs = list(range(100))
    assert sample(pairs, 50) == sample(pairs, 50) and len(set(sample(pairs, 50))) == 50
    assert sorted(sample(pairs[:10], 50)) == pairs[:10]
