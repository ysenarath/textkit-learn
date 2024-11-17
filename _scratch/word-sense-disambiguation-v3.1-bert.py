from typing import List

import faiss
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Lemma, Synset
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Download WordNet if not already downloaded
nltk.download("wordnet")


def get_wordnet_examples() -> pd.DataFrame:
    """Retrieve all examples from WordNet with their associated words/phrases"""
    examples = []
    # Iterate through all synsets in WordNet
    all_sunsets: List[Synset] = list(wn.all_synsets())
    for synset in tqdm(list(all_sunsets), desc="Collecting examples"):
        # Get examples for the synset
        for example in synset.examples():
            # Get all lemma names (words) associated with this synset
            lemma: Lemma
            for lemma in synset.lemmas():
                examples.append({
                    "lemma": lemma.name(),
                    "example": example,
                    "pos": synset.pos(),
                    "synset": synset.name(),
                })
    return pd.DataFrame(examples)


def create_prompt(
    row: dict = None, /, *, lemma: str = None, example: str = None
) -> str:
    """Create a prompt in the specified format"""
    lemma = row["lemma"] if lemma is None else lemma
    example = row["example"] if example is None else example
    return f'Given the word "{lemma}", here is an example of its usage: "{example}"'


def create_prompts(df: pd.DataFrame) -> List[str]:
    """Create prompts in the specified format"""
    return df.apply(create_prompt, axis=1).tolist()


def generate_embeddings(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Generate embeddings for a list of texts using sentence-transformers"""
    # Generate embeddings
    embeddings = []
    # Process in batches to manage memory
    kwargs = {"desc": "Generating embeddings"}
    for i in tqdm(range(0, len(texts), batch_size), **kwargs):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    # Concatenate all embeddings
    all_embeddings = torch.cat(embeddings, dim=0)
    return all_embeddings.cpu().numpy()


def create_vector_db(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    # Get dimensionality of embeddings
    d = embeddings.shape[1]
    # Create FAISS index
    # Using L2 (Euclidean) distance for similarity search
    index = faiss.IndexFlatL2(d)
    # Add vectors to the index
    print("Adding vectors to FAISS index...")
    index.add(embeddings.astype(np.float32))  # FAISS requires float32
    print(f"Created FAISS index with {index.ntotal} vectors of dimension {d}")
    return index


def main():
    model_name_or_path = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name_or_path)
    # Get examples and create prompts
    print("Collecting WordNet examples...")
    df = get_wordnet_examples()
    prompts = create_prompts(df)
    # Generate embeddings
    print("\nGenerating embeddings...")
    embeddings = generate_embeddings(prompts, model)
    # Save embeddings and metadata
    # np.save("wordnet_embeddings.npy", embeddings_np)
    # df.to_csv("wordnet_metadata.csv", index=False)
    print("\nResults saved:")
    print("- Embeddings shape:", embeddings.shape)
    print("- Metadata saved to: wordnet_metadata.csv")
    print("- Embeddings saved to: wordnet_embeddings.npy")
    # Create FAISS index
    index = create_vector_db(embeddings)
    # Number of nearest neighbors to retrieve
    k = 5
    # Example input texts
    input_texts = [
        # robbed
        ("robbed", "The bank was robbed last night."),
        # bank
        ("bank", "The bank was robbed last night."),
        ("bank", "I sat by the river bank and read a book."),
        # bass
        ("bass", "The bass line of the song was very catchy."),
        ("bass", "The bass singer had a deep voice."),
        # bat
        ("bat", "The bat flew around the room."),
        ("bat", "The baseball player swung the bat."),
    ]
    input_prompts = [
        create_prompt(lemma=lemma, example=text) for lemma, text in input_texts
    ]
    input_np = generate_embeddings(input_prompts, model)
    D, I = index.search(input_np, k)
    print("\nNearest neighbors:")
    for i, input_text in enumerate(input_texts):
        print(f"\nInput text: {input_text}")
        for j in range(k):
            index = I[i, j]
            example = df.iloc[index]
            print(f"Nearest neighbor {j + 1}:")
            print(f"  - Example: {example['example']}")
            print(f"  - Synset: {example['synset']}")
            print(f"  - POS: {example['pos']}")


if __name__ == "__main__":
    main()
