import numpy as np
from RNN import RNNModel
from RNN_utils import SGD, one_hot_encoding

np.random.seed(1)

person_names = open('person_names.txt', 'r').read()
person_names= person_names.lower()
characters = list(set(person_names))

character_to_index = {character:index for index,character in enumerate(sorted(characters))}
index_to_character = {index:character for index,character in enumerate(sorted(characters))}

with open("person_names.txt") as f:
    person_names = f.readlines()

person_names = [name.lower().strip() for name in person_names]
np.random.shuffle(person_names)

print(person_names[:5])

num_epochs = 200001
input_dim = 27
output_dim = 27
hidden_dim = 50

# initialize
model = RNNModel(input_dim, output_dim, hidden_dim)
optim = SGD(lr=0.01)
costs = []

# Training
for epoch in range(num_epochs):
    
    # create X inputs, Y labels
    index = epoch % len(person_names)
    X = [None] + [character_to_index[ch] for ch in person_names[index]] 
    Y = X[1:] + [character_to_index["\n"]]

    # hot encoding
    X = one_hot_encoding(X, input_dim)
    Y = one_hot_encoding(Y, output_dim)
    
    # steps
    model.forward(X)
    cost = model.loss(Y)
    model.backward()
    # clip grads
    model.clip(clip_value=1)
    # opt
    model.optimize(optim)

    if epoch % 10000 == 0:
        print ("Cost after iteration %d: %f" % (epoch, cost))
        costs.append(cost)

        print('Names created:', '\n')
        for i in range(4):
            name = model.generate_names(index_to_character)
            print(name)
