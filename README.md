# Talk to your PDF
Chat with your PDF using Python, Langchain and ollama

## Installation

### Prerequisites

1. Python 3.11 or higher
1. Git
1. ollama installed on your system or in a local network

### Steps
1. Clone the repository
    ```bash
    git clone https://github.com/margitantal68/rag_inhouse_llm
    ```

1. Navigate to the project directory
    ```bash
    cd rag_inhouse_llm
    ```

1. Create and activate a virtual environment
    * On Linux/macOS:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

    * On Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```


## Usage


1. Change the **base_url** and the **model** in the **main.py** file to the URL of your ollama instance and your model.
    ```python
    llm = OllamaLLM(base_url="http://192.168.11.102:11500", model="mistral-nemo:12b-instruct-2407-q8_0") 
    ```
1. Run the project:
    ```bash
    python main.py
    ```

