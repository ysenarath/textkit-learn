import marisa_trie


def __getattr__(name):
    return getattr(marisa_trie, name)
