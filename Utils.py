import numpy as np
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

class VectorUtils:
    def angle_between_vectors(v1, v2):
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        cos_angle = dot_product / (norm_v1 * norm_v2)
        angle = np.arccos(cos_angle)
        angle_degrees = np.degrees(angle)    
        return angle_degrees, cos_angle
    
class ModelUtils:
    def average_angle(model, corpus):
        # calculate average angle between every combination of words in the corpus
        angles = []
        for i in corpus:
            wordi = i[0]
            for j in corpus:
                wordj=j[0]
                if wordi!=wordj:
                    angles.append(VectorUtils.angle_between_vectors(model.wv[wordi], model.wv[wordj])[0])
        average_angle = sum(angles)/len(angles)
        print("Average angle: ", average_angle)
        return average_angle
    
    def train_model(corpus, dimensions):
        # Train Word2Vec model
        model = Word2Vec(corpus, vector_size=dimensions, window=5, min_count=1, sg=1)
        return model
    
    # get distinct words in a corpus
    def corpus_vocab(corpus):
        return np.unique([word for sublist in corpus for word in sublist])
    
    def prepare_word_plots(vocab, model, dimensions_to_plot=[0,1]):
        xs = []
        ys = []
        annotations = []
        for i, word in enumerate(vocab):
            v = model.wv[word]
            x = v[dimensions_to_plot[0]]
            y = v[dimensions_to_plot[1]]
            xs.append(x)
            ys.append(y)
            annotations.append(word)

        data = {
            "xs": xs,
            "ys": ys,
            "annotations": annotations             
        }
        return data
    def print_word_differences(model, word1, word2):
        print(f"\n {word1} - {word2}")
        angle, similarity = VectorUtils.angle_between_vectors(model.wv[word1], model.wv[word2])
        print("Angle between: ", angle)
        print("Cosine Similarity: ", similarity)
    
class GraphUtils:
    
    def render_multiple_graphs(data_list, columns=3):
       # Determine the layout based on the number of graphs
        n_graphs = len(data_list)
        rows = (n_graphs + columns - 1) // columns  # Calculate number of rows needed
        
        # Create a figure with subplots
        fig, axs = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
        for i, data in enumerate(data_list):
            ax = fig.add_subplot(rows,columns,i+1)                            
            if "title" in data:
                ax.title.set_text(data["title"])
            ax.scatter(data["xs"], data["ys"])
            for i, word in enumerate(data["annotations"]):
                ax.text(data["xs"][i]+0.01, data["ys"][i]+0.01, word, fontsize=12, ha='center')
            
            #add lines through origin
            ax.axhline(y=0, color='r', linestyle='-')
            ax.axvline(x=0, color='r', linestyle='-')

        # Adjust layout
        plt.tight_layout()
        plt.show()     