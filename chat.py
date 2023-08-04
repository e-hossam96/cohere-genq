"""Chat with the custom data.

Send a question to the Cohere model and answer it using the \
    retrieved contexts from Pinecone index.
"""


import json
import cohere
import pinecone
from pprint import pprint as pp
from langchain import PromptTemplate


def llm_chat(api, index, question, prompt_temp,
             embedding_model='embed-english-v2.0'):
    """Chat with the stored data.

    Embed the question using the Cohere model, then retrieve the \
        most relevant text snippets.
    Use the retrieved snippets as contexts for the Cohere model to \
        answer from.
    """
    xq = api.embed(texts=[question], model=embedding_model).embeddings
    result = index.query([xq], top_k=5, include_metadata=True)
    contexts = '\n\n'.join([r['metadata']['text'] for r in result['matches']])
    prompt = prompt_temp.format(context=contexts, question=question)
    answer = api.generate(prompt, temperature=0.5,
                          model='command', max_tokens=128)
    return answer.generations[0].text


def main(tokens_path):
    """Construct the main chat pipeline."""
    template = '''Answer the question based on the context below. If the
    question cannot be answered using the information provided answer
    with ### I don't know ###.

    ###

    context: {context}

    ###

    question: {question}
    answer:'''
    prompt_template = PromptTemplate(input_variables=['context', 'question'],
                                     template=template)

    with open(tokens_path) as f:
        access_tokens = json.load(f)

    api = cohere.Client(access_tokens['cohere']['api_key'])
    pinecone.init(api_key=access_tokens["pinecone"]["api_key"],
                  environment=access_tokens["pinecone"]["env"])
    index = pinecone.Index("cohere-gqa")

    print('Lets Chat ...')
    again = True
    while again:
        question = input('What do you have in mind?:\n>>> ')
        pp(llm_chat(api, index, question, prompt_template))
        again = input('Any more question? (y/n)\n>>> ') == 'y'
    return None


if __name__ == '__main__':
    tokens_path = './access_tokens.json'
    print("Loading Setups ...")
    main(tokens_path)
