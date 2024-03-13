from langchain_community.embeddings import BedrockEmbeddings
from langchain.evaluation import load_evaluator, EvaluatorType

def main():
    #init embeddings
    embeddings = BedrockEmbeddings(
        credentials_profile_name="sre-sandbox-genai", region_name="us-east-1"
    )

    #create embedding
    vector = embeddings.embed_query("apple")

    #describe embedding
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    #compare embedding
    evaluator = load_evaluator(EvaluatorType.PAIRWISE_EMBEDDING_DISTANCE)
    words = ("apple", "iphone")

    x = evaluator.evaluate_string_pairs(prediction=words[0], prediction_b=[1])
    print(f"Comparing ({words[0]}, {words[1]}): {x}")


if __name__ == "__main__":
    main()


