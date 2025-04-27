import pickle

from tklearn.kb import KnowledgeBase


def test_pickle():
    kb = KnowledgeBase("wiktionary")
    # Test pickling and unpickling the KnowledgeBase
    pickle.loads(pickle.dumps(kb))
    # Test pickling and unpickling the KnowledgeBase with a different protocol
    pickle.loads(pickle.dumps(kb, protocol=pickle.HIGHEST_PROTOCOL))


if __name__ == "__main__":
    test_pickle()
    print("Pickling test passed.")
