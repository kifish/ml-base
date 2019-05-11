import matplotlib.pyplot as plt

loss = []
with open('./a.train.log','r',encoding='utf8') as f:
    f.readline()
    for line in f.readlines():
        loss.append(float(line.split(',')[1][1:]))

val_loss = []
with open('./a.valid.log','r',encoding='utf8') as f:
    f.readline()
    for line in f.readlines():
        val_loss.append(float(line.split(',')[1][1:]))


epochs = range(1, len(loss) + 1)
plt.figure(0)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'g', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('transformer_Training_and_validation_loss.png')


