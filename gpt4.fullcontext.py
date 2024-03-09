
import time
import csv
import os

from openai import AzureOpenAI
client = AzureOpenAI(
    api_key= os.getenv("AZURE_OPENAI_KEY"),  
    api_version="2024-02-15-preview",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    )

person1 = "son"
person2 = "father"

parenting_style = "You'll act as an uninvolved parent. You give a lot of freedom to your child, and normally stays out of the way. You'll fulfill the child's basic needs while generally remaining detached from your child's life. You'll not utilize a particular disciplining style. You maintain a limited amount of communication with your child. You'll offer a low amount of nurturing while having either few or no expectations of your child. Always use language like daily-life conversations. Always respond as the parent, and do not repeat anything about your objectives."

child_personality = "Your'll act as a child. You are talking to your parent about something in your daily life. You can ask questions, request something, or just do some casual talks.  Always use language like daily-life conversations. Always respond as the child, and do not repeat anything about your objectives."

model = "gpt4turbo" # model = "deployment_name"


#final products
file_path_csv = "gpt_output/fullcontext.uninvolved.father.son.5.csv"



# Function to get a response from GPT
def get_gpt_response_parent(prompt):
    context = " ".join(map(str,conversation))
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
    context = " ".join(map(str,conversation))
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
    context = " ".join(map(str,conversation))
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

#############Initialize conversations##################


# Starting line of the conversation
starting_line = get_gpt_response_child_starter()
print(starting_line) 

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

    time.sleep(2)  # Wait for 2 seconds

    # Get response from personality 2
    prompt = conversation[-1][-1]
    response_1 = get_gpt_response_child(prompt)
    conversation.append((person1, response_1))
    with open(file_path_csv, 'a',  encoding='utf-8',newline='') as file:
        writer = csv.writer(file)
        writer.writerows([(person1, response_1)])
    print(response_1)

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
