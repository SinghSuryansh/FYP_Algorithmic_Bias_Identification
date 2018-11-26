#Importing the required libraries:
import nltk, io, os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import warnings
#Ignoring the warnings
warnings.filterwarnings('ignore')

#Setting the path to the input files:
path = './Data/'

#Initializing an empty list to store the input for the word2vecmodel:
input = []
#To keep track of progress:
i = 0

#Processing the files:
for root, directories, files in os.walk(path):
	for filename in files:

		filepath = os.path.join(root, filename)

		f = open(filepath,"r",encoding="utf-8")

		#Reading in the content of the file:
		content = f.read()

		#Converting the text into lower case:
		content = content.lower()

		#replacing certain words within the text:
		content = content.replace('south africa', 'southafrica')
		content = content.replace('southafrican', 'southafrican')
		content = content.replace('ivory coast', 'ivorycoast')
		content = content.replace('sierra leone', 'sierraleone')
		content = content.replace('sierra leonean', 'sierraleonean')
		content = content.replace('british somaliland','britishsomaliland')
		content = content.replace('afar people', 'afarpeople')
		content = content.replace('amhara people','amharapeople')
		content = content.replace('anlo-ewe','anloewe')
		content = content.replace('ewe people', 'ewepeople')
		content = content.replace('ashanti people','ashantipeople')
		content = content.replace('asante people', 'asantepeople')
		content = content.replace('dinka people','dinkapeople')
		content = content.replace('fang people','fangpeople')
		content = content.replace('massai people','massaipeople')
		content = content.replace('san people', 'sanpeople')
		content = content.replace('saan people', 'saanpeople')
		
		#Splitting the text into tokens:
		tokens = word_tokenize(content)

		#removing the punctuation, numbers and special characters:
		tokens = [word for word in tokens if word.isalpha()]

		#removing the stopwords:
		stopwords_nltk_en = set(stopwords.words('english'))
		tokens = [word for word in tokens if word not in stopwords_nltk_en]

		#Generating the input:
		input.append(tokens)
		i+=1
		print("Complete",i)


#Gensim model:
print("Training the model")
#Importing the word2vec model:
from gensim.models import Word2Vec
model = Word2Vec(input, size=100, window=10, min_count=1)
print("Model Trained")
#Saving the model:
model.save("Bias.model")
#Saving as a bin file:
from gensim.models import KeyedVectors
model.wv.save_word2vec_format('Bias.bin',binary=True)

