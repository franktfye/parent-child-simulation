# parent-child-simulation
 Simulating Family Conversations using LLMs: Demonstration of Parenting Styles

This repository contains the python scripts for simulating conversations between two LLM powered agents. 
As demonstrated in our paper, we simulated conversations between a child and a parent in various conditions. 
Details can be found in the below paper:

## Citation

```
@article{}
```

##Run
To run the scripts, simply download them and execute them in a python environment. Each .py file is corresonding to one model setting. You can change the personality of the agent (e.g., parenting style) in the script.
We tested them in python 3.11.
For the Ollama server, you need to download Ollama <a href="https://ollama.com/" target="_blank"> from their website </a> and pull the model to your computer to run the script locally. 

We used the mixtral:8x7b-instruct-v0.1-q5_K_M in our simulations.
```bash
ollama pull mixtral:8x7b-instruct-v0.1-q5_K_M
ollama serve
```

For the GPT-4 model, we accessed API via AzureOpenAI. You need to setup the api key and endpoint in system environmental variables before you execute these scripts. You can also change it to official OpenAI API if you like.
