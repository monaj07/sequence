import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import pdb

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

# By deriving a set from `raw_text`, we deduplicate the array
vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))
print(data[:5])

print(torch.__version__)


class CBOW(nn.Module):

    def __init__(self, vocab_size, context_size, embedding_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear1 = nn.Linear(context_size * embedding_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view(1, -1)
        x = self.linear1(embeds)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=1)
        return x

# create your model and train.  here are some functions to help you make
# the data ready for use by your module


def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.FloatTensor(idxs).long()


model = CBOW(vocab_size, CONTEXT_SIZE*2, 10)
optimizer = optim.SGD(model.parameters(), lr = 0.1)
LOSS = nn.NLLLoss()

for epoch in range(10):
	total_loss = 0
	for sample in data:
		inp = make_context_vector(sample[0], word_to_ix)
		tar = torch.FloatTensor([word_to_ix[sample[1]]]).long()
		model.zero_grad()
		output = model(Variable(inp))
		loss = LOSS(output, Variable(tar))
		loss.backward()
		optimizer.step()
		total_loss += loss.data.numpy().item()
	print('epoch {}, loss = {}.'.format(epoch+1, total_loss))
		

