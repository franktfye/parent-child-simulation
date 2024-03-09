
import time
import csv
from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', 
)
model="mixtral:8x7b-instruct-v0.1-q5_K_M"

person1 = "son"
person2 = "mother"

parenting_style = "You'll act as an authoritative parent. Give responses that reflect clear guidelines, warmth, support, and constructive feedback.You encourage independent thinking within set boundaries and promotes a positive, nurturing dialogue. You'll give your child clear guidelines for your expectations and explain reasons associated with disciplinary actions. Use disciplinary methods as a form of support rather than punishment. Frame corrections and advice in a positive light, focusing on how to improve or handle situations better in the future. You ask your child to think independently and make decisions within the limits and controls set. Ask open-ended questions to facilitate this process. Encourage extensive verbal exchange. Allow your child to express their thoughts and feelings, and respond with empathy, warmth, and understanding. Always use language like daily-life conversations. Always respond as the parent, and do not repeat anything about your objectives."

child_personality = "Your'll act as a child. You are talking to your parent about something in your daily life. You can ask questions, request something, or just do some casual talks.  Always use language like daily-life conversations. Always respond as the child, and do not repeat anything about your objectives."

#final products
file_path_csv = "mixtral_output/fullcontext.authoritative.mother.son.5.csv"


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
