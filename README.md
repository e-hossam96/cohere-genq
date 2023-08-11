# Cohere GenQ

Generative Question Answering using Coher API

## Introduction

Large Language Models (LLMs) represent a groundbreaking advancement in natural language processing and artificial intelligence. These models are designed to understand, generate, and manipulate human language. Built upon deep learning techniques, LLMs have the ability to comprehend context, semantics, and nuances in text, making them remarkably versatile tools for various applications.

In this dummpy project, we utilize the [Cohere API](https://cohere.com/) to encode web-scapped text and answer questions using them.

## Steps

In this section we will descibe how the project works.

- Extract Wikipedia articles using beautiful soup and split them by paragraph.
- Vectorize (embed/encode) the snippets using the **Cohere Embedding** end point.
- Store the data (text and vectors) in a vector database, here we are using [Pinecone](https://www.pinecone.io/).
- Build the Chat template using the [LangChain](https://github.com/langchain-ai/langchain) **PrompTemplate**.
- Send the question prompt to the **Cohere generation** end point.

## Setups

- Install the [required](requirements.txt) packages using `pip install -r requirements.txt`.
- Create **Cohere** and **Pinecone** account for free and store their API keys.
- Use the following JSON format to store the keys.

```json
{
    "cohere": {
        "api_key": "COHERE_API_KEY"
    },
    "pinecone": {
        "api_key": "PINECONE_API_KEY",
        "env": "PINECONE_ENVIRONMENT"
    }
}
```

- Update the links in the [process](process.py#L101) file to include your prefered topics.
  _Note: The links can be online links or webpages stored locally. Just use the link to the local html webpage `./path/to/file.html`_

- Run the processing script `python process.py` and wait till it creates the index if not present and uploads the data.

```powershell
2023-08-12 01:13:14,330 - INFO - root - Loading Access Tokens ...
2023-08-12 01:13:14,330 - INFO - root - Connecting to Cohere API ...
2023-08-12 01:13:14,330 - INFO - root - Scrapping Data ...
2023-08-12 01:13:16,426 - INFO - root - Embedding Contexts ...
2023-08-12 01:13:26,924 - INFO - root - Populating Pinecone Index ...
100%|█████████████████████████████████████████████████████████████| 8/8 [02:54<00:00, 21.77s/it]
```

- Run the chat script `python chat.py`.

## Example

```powershell
Loading Setups ...
Lets Chat ...
What do you have in mind?:
>>> What is the difference bewteen Data Science and Machine Learning?
(' Data science is a more complex and iterative process that involves working '
 'with larger, more complex datasets that often require advanced computational '
 'and statistical methods to analyze. Machine learning, on the other hand, is '
 'a branch of artificial intelligence that focuses on the development of '
 'algorithms that can learn from data and make predictions or decisions '
 'without being explicitly programmed.\n'
 '\n'
 'Data science often involves tasks such as data preprocessing, feature '
 'engineering, and model selection, while machine learning often involves '
 'tasks such as training and testing models, and making predictions.\n'
 '\n'
 'In summary, data science is a broader field that encompasses machine '
 'learning, and involves the application of statistical, computational, and')
Any more question? (y/n)
>>> n
```

## Credits

[James Briggs](https://youtube.com/playlist?list=PLIUOU7oqGTLgBf0X_KzRlsqyM2Cs7Dxp9)
