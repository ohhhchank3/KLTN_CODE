
from langchain.document_loaders.unstructured import UnstructuredFileLoader


def pdf2text(filepath):
    loader = UnstructuredFileLoader(file_path=filepath)
    pages = loader.load_and_split()

    return pages


if __name__ == "__main__":
    loader = pdf2text(filepath="./test/demo1.pdf")
    print(loader[0])