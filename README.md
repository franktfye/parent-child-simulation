# parent-child-simulation
 Simulating Family Conversations using LLMs: Demonstration of Parenting Styles

This repository contains Python scripts for simulating conversations between two LLM-powered agents.

As demonstrated in our paper, we simulated conversations between a child agent and a parent agent under various conditions. The results of these simulations have been uploaded to the output folders within this repository.

Further details can be found in the paper referenced below:

## Citation

```
@misc{ye2024simulating,
      title={Simulating Family Conversations using LLMs: Demonstration of Parenting Styles}, 
      author={Frank Tian-fang Ye and Xiaozi Gao},
      doi={https://doi.org/10.48550/arXiv.2403.06144},
      year={2024},
      eprint={2403.06144},
      archivePrefix={arXiv},
      primaryClass={cs.CY}
}
```

## Run
These are simple scripts that can be run with minimal setup.

Before running the scripts, ensure that you have the OpenAI package installed in your Python environment.

The simulation settings can be modified directly within each script. You can customize the personalities of the agents, the number of conversation exchanges, output file names, and other parameters.

To run the scripts, simply download them and execute them in a Python environment. Each .py file corresponds to a specific model setting. You can change the personality of the agent (e.g., parenting style) within the script itself.

We have tested these scripts using Python 3.11.

For the Ollama server, you need to download Ollama <a href="https://ollama.com/" target="_blank"> from their website </a> and pull the model to your computer to run the script locally. 

We used the mixtral:8x7b-instruct-v0.1-q5_K_M in our simulations.
```bash
ollama pull mixtral:8x7b-instruct-v0.1-q5_K_M
ollama serve
```

To access the GPT-4 model, we used the Azure OpenAI API. Before executing these scripts, you need to set up the API key and endpoint in your system's environment variables. Alternatively, you can modify the scripts to use the official OpenAI API if you prefer.

Ensure that you have deployed the necessary models and updated the scripts with the correct model names before running them.

To set up the environment variables for Azure OpenAI, follow these steps:

Obtain your API key and endpoint from your Azure OpenAI resource.
Open your system's environment variables settings.
Add a new environment variable named OPENAI_OPENAI_KEY and set its value to your API key.
Add another environment variable named AZURE_OPENAI_ENDPOINT and set its value to your Azure OpenAI endpoint.
Once the environment variables are set up and the model names are correctly specified in the scripts, you can proceed to execute them in your Python environment.
