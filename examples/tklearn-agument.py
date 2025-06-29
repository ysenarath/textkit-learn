from tklearn.kb import KnowledgeBase

kb = KnowledgeBase("wiktionary")

text = "She's a pure Oreo. You know, like the cookie, black outside and white inside."

for mention in kb.extract_mentions(text):
    start, end = mention.span
    mention_text = text[start:end]
    start_tag = "<m>"
    end_tag = "</m>"
    text_annotated = (
        text[:start] + start_tag + mention_text + end_tag + text[end:]
    )
    print(text_annotated)
