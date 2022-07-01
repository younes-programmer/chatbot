import random
import json
import torch
from model import NeuralNetwork
from pre_processing import *

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')
intents = json.loads(open('intents.json').read())
my_file = "data.pth"
data = torch.load(my_file)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model-state"]
model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "IT_CRMEF"


# print("Enter your name ")
def get_response(msg):
    sentence = tokenize(msg)
    sen_digits = bag_of_words(sentence, all_words)
    sen_digits = sen_digits.reshape(1, sen_digits.shape[0])
    sen_digits = torch.from_numpy(sen_digits).to(device)

    output = model(sen_digits)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probabilities_prediction = torch.softmax(output, dim=1)
    probability = probabilities_prediction[0][predicted.item()]
    if probability.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return f"{random.choice(intent['responses'])}"
    else:
        return f"Excuse me i don't understand can you repeat again ?"


if __name__ == "__main__":
    student_name = input('please enter your name ')
    print(f"welcome {student_name} :) 'type q to quit'")
    while True:
        msg = input(f'{student_name} : ')
        if msg == "q":
            break
        response = get_response(msg)
        print(response)
