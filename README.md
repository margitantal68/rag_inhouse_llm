# Talk to your PDF
Chat with your PDF using Python, Langchain and OpenAI

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

This project requires an OpenAI API key. Follow these steps to set it up:

1. Change the **base_url** in the **main.py** file to the URL of your ollama instance.
    

1. Run the project:
    ```bash
    python main.py
    ```

