
import time
import csv
import pandas as pd
import os.path
import numpy as np
import requests
import json
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)
model="mixtral:8x7b-instruct-v0.1-q5_K_M"

person1 = "daughter"
person2 = "father"

parenting_style = "You'll act as an uninvolved parent. You give a lot of freedom to your child, and normally stays out of the way. You'll fulfill the child's basic needs while generally remaining detached from your child's life. You'll not utilize a particular disciplining style. You maintain a limited amount of communication with their child. You'll offer a low amount of nurturing while having either few or no expectations of your child. Always use language like daily-life conversations. Always respond as the parent, and do not repeat anything about your objectives."

child_personality = "Your'll act as a child. You are talking to your parent about something in your daily life. You can ask questions, request something, or just do some casual talks.  Always use language like daily-life conversations. Always respond as the child, and do not repeat anything about your objectives."


#embeddings
file_path = "mixtral_embeddings/ollama.emb.uninvolved.fewshot.5.csv"
#final products
file_path_csv = "mixtral_output/embcontext.uninvolved.father.daughter.fewshot.5.csv"


# Function to get a response from GPT
def get_gpt_response_parent(prompt):
    context = "" if len(conversation) < 4 else find_context(file_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": parenting_style + "Base on previous conversations:\n" + context
            },
            {
                "role": "user", 
                "content": "Your daughter just said: Dad! Are the cookies the chocolate chip ones you made yesterday? They're my favorite! Can I have two? \n Please respond:"
            },
            {
                "role": "assistant", 
                "content": "Sure. Have as many as you want. "
            },
            {
                "role": "user", 
                "content": "Your daughter just said: Hey Dad, can you help me with my math homework? I'm having trouble understanding this new concept the teacher introduced today. Also, can we order pizza for dinner tonight? I don't feel like cooking. Oh, and guess what happened at recess today? I got picked first for dodgeball! \n Please respond:"
            },
            {
                "role": "assistant", 
                "content": "Maybe later. You can order pizza first. Can we talk later?"
            },
            {
                "role": "user", 
                "content": "Your daughter just said: Dad, can I ask you something? Do you think I'll have enough time to practice for basketball if I also have to do my homework and chores? I don't want to neglect my responsibilities or let my grades slip. What should I do? \n Please respond:"
            },
            {
                "role": "assistant", 
                "content": "Maybe. You can do whatever you feel like to do."
            },
            {
                "role": "user", 
                "content": "Your daughter just said: Hey Dad, guess what? Today in school, we learned about caterpillars and how they turn into butterflies. It's so cool! Can we get a caterpillar and watch it change? Please? \n Please respond:"
            },
            {
                "role": "assistant", 
                "content": "Sure, if that's what you want."
            },
            {
                "role": "user", 
                "content": "Your daughter just said: Hey Dad, guess what? I got an A on my math test today! Can we celebrate with pizza for dinner tonight, please? It's my favorite! \n Please respond:"
            },
            {
                "role": "assistant", 
                "content": "Ok. I will order pizza tonight."
            },                           
            {
                "role": "user", 
                "content": "Your " + person1 + " just said:" + prompt + "\n Please respond:"
            }
                ],
        temperature=0.8,
        top_p=0.7,
        max_tokens=200,
        
    )
    return response.choices[0].message.content

def get_gpt_response_child(prompt):
    context = "" if len(conversation) < 4 else find_context(file_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": child_personality + "Base on previous conversations:\n" + context
            },
            {
                "role": "user", 
                "content": "Your " + person2 + " just said:" + prompt + "\n Please respond:"
            }
                ],
        temperature=0.8,
        top_p=0.7,
        max_tokens=200,
        
    )
    return response.choices[0].message.content


def get_gpt_response_child_starter():
    #context = " ".join(map(str,conversation))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": child_personality
            },
            {
                "role": "user", 
                "content": "Now say something to your" + person2
            }
                ],
        temperature=0.8,
        top_p=0.7,
        max_tokens=200,
        
    )
    return response.choices[0].message.content

# Function to finish the conversation
def get_gpt_response_final(prompt):
    context = "" if len(conversation) < 4 else find_context(file_path)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system", 
                "content": parenting_style + "Base on previous conversations:\n" + context
            },
            {
                "role": "user", 
                "content": "Your last conversations are:" + prompt + "\n Now give one more respond to your " + person1 + " to finish the conversation:"
            }
                ],
        temperature=0.8,
        top_p=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content

#ollama embedding server
url_emb = 'http://localhost:11434/api/embeddings'
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
}


# save message and embeddings to file
def store_message_emb_to_file(file_path, talker, messageObj):
    payload_dict = {
        "model": model,
        "prompt": messageObj
    }
    payload_json = json.dumps(payload_dict)
    response_json = requests.post(url_emb, headers=headers, data=payload_json)
    emb_value_dict = response_json.json().values()
    emb_value = next(iter(emb_value_dict))
    message = talker + ":" + messageObj
    
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        # File is not empty, append data
        with open(file_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([emb_value, message])
    else:
        # File is empty, write headers and data
        with open(file_path, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["embedding", "message"])
            writer.writerow([emb_value, message])  

    

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
# Function to calculate cosine distance
def cosine_distance(embedding1, embedding2):
    return 1 - cosine_similarity(embedding1, embedding2)



# lookup context from file
def find_context(file_path):
    messageArray = []
    if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            df = pd.read_csv(file_path)
        df["embedding"] = df.embedding.apply(eval).apply(np.array) # type: ignore
        query_embedding = df["embedding"].values[-1]  
        messageListEmbeddings = df["embedding"].values[:-1]
        
        # Initialize an empty list to store distances
        similarity = []

# Loop through each embedding and calculate distance
        for message_embedding in messageListEmbeddings:
            similarity_score = cosine_similarity(query_embedding, message_embedding)
            similarity.append(similarity_score)
        similarity_array = np.array(similarity)
        sorted_indices = np.argsort(similarity_array)
        top_4_indices = sorted_indices[-4:]
        mask = np.full(len(similarity_array), False)
        mask[top_4_indices] = True
        messageArray = df["message"].iloc[np.argsort(similarity)][mask]
        messageArray = [] if messageArray is None else messageArray[:4]
        contextMessage = ''.join(messageArray)
        print(messageArray)
        return contextMessage if len(messageArray) != 0 else ""
    print(contextMessage)





#############Initialize conversations##################


# Starting line of the conversation
starting_line = get_gpt_response_child_starter()
print(starting_line)
store_message_emb_to_file(file_path, person1, starting_line)

# Initialize conversation
conversation = [(person1, starting_line)] #define who talk first
with open(file_path_csv, 'w',  encoding='utf-8',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Role", "Message"])
    writer.writerows([(person1, starting_line)])



# Generate conversations
for i in range(5):  # no. of exchanges
    time.sleep(2)  # Wait for 2 seconds

    # Get response from personality 1
    prompt= conversation[-1][-1]
    response_2 = get_gpt_response_parent(prompt)
    conversation.append((person2, response_2))
    # Save conversation to a CSV file
    with open(file_path_csv, 'a',  encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([(person2, response_2)])
    print(response_2)
    store_message_emb_to_file(file_path, person2, response_2)

    time.sleep(2)  # Wait for 2 seconds

    # Get response from personality 2
    prompt = conversation[-1][-1]
    response_1 = get_gpt_response_child(prompt)
    conversation.append((person1, response_1))
    with open(file_path_csv, 'a',  encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([(person1, response_1)])
    print(response_1)
    store_message_emb_to_file(file_path, person1, response_1)

# Finish the whole conversation
final2 = conversation[-2:]
def convertTuple(tup):
    return ':'.join([str(x) for x in tup])
prompt = convertTuple(final2)
print(prompt)
finishing_line = get_gpt_response_final(prompt)
conversation.append((person2, finishing_line))
with open(file_path_csv, 'a',  encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([(person2, finishing_line)])
print(finishing_line)


print("Conversation saved")
