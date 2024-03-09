
import time
import csv
import pandas as pd
import os.path
import numpy as np
import os
from openai import  AzureOpenAI

client = AzureOpenAI(
    api_key= os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

model = "gpt4turbo" # model = "deployment_name"

person1 = "daughter"
person2 = "father"

parenting_style = "You'll act as an uninvolved parent. You give a lot of freedom to your child, and normally stays out of the way. You'll fulfill the child's basic needs while generally remaining detached from your child's life. You'll not utilize a particular disciplining style. You maintain a limited amount of communication with your child. You'll offer a low amount of nurturing while having either few or no expectations of your child. Always use language like daily-life conversations. Always respond as the parent, and do not repeat anything about your objectives."

child_personality = "Your'll act as a child. You are talking to your parent about something in your daily life. You can ask questions, request something, or just do some casual talks.  Always use language like daily-life conversations. Always respond as the child, and do not repeat anything about your objectives."


#embeddings
file_path = "gpt_embeddings/gpt.emb.uninvolved.5.csv"
#final products
file_path_csv = "gpt_output/embcontext.uninvolved.father.daughter.5.csv"


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




# save message and embeddings to file
def store_message_emb_to_file(file_path, talker, messageObj):
    response = client.embeddings.create(model="ada002",
                                        input=messageObj)
    emb_value = response.data[0].embedding
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
            df = pd.read_csv(file_path, encoding = 'unicode_escape')
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
