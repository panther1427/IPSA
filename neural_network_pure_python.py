import gzip
import matplotlib.pyplot as plt
import json
import random
from matplotlib.animation import FuncAnimation

'''
Made by:
    202204624 Lucas Bjerre Rasmussen
'''

#### - PART 1 - ####
def read_labels(filename):
    """Return a list of labels from a given file.
    
    Arguments:
    filename -- The absolute or relative path to the file
    
    Raises exception if the magic number is not 2049
    """
    with gzip.open(filename=filename, mode="rb") as file:
        file = file.read()
    if int.from_bytes(file[:4], "big") != 2049:
        raise Exception("Wrong file")
    
    # Add 8, because file uses 8 bytes to track image amount and magic number
    return [file[i] for i in range(8, int.from_bytes(file[4:8], "big") + 8)]
        

def read_images(filename):
    """Return a list of images in matrix form
    
    Arguments:
    filename -- absolute or relative path to the file,
                from which images are to be loaded
    
    Raises exception if the magic number is not 2051
    """
    with gzip.open(filename=filename, mode="rb") as file:
        file = file.read()

    if int.from_bytes(file[:4], "big") != 2051:
        raise Exception("Wrong file")
    
    images_count = int.from_bytes(file[4:8], "big")
    images = [[[file[16 + picture * (28 ** 2) + row * 28 + column] 
                for column in range(0, 28)] 
                for row in range(0, 28)] 
                for picture in range(0, images_count)]
    
    return images
    

#### - PART 2 - ####
def linear_load(file_name):
    """Return a network = (A, b)"""
    with open(f"{file_name}") as file:
        return json.load(file)


def linear_save(file_name, network):
    """Save a network = (A, b)"""
    with open(file_name, "w") as outfile:
        json.dump(network, outfile)


def image_to_vector(image):
    """Convert image matrix to vector and rescale pixel values"""
    return [pixel/255 for row in image for pixel in row]
    

def add(U, V):
    """Return the sum of two vectors"""
    assert len(U) == len(V)
    return [U[i] + V[i] for i in range(len(U))]


def sub(U, V):
    """Return the difference of two vectors"""
    assert len(U) == len(V)
    return [U[i] - V[i] for i in range(len(U))]


def scalar_multiplication(scalar, V):
    """Return the product of a scalar and a vector"""
    return [scalar * V[i] for i in range(len(V))]


def multiply(V, M):
    """Return the matrix multiplication of a vector and a matrix"""
    assert len(V) == len(M)    
    return [sum(V[i] * M[i][j] for i in range(len(V)))    # ROWS
                               for j in range(len(M[0]))] # COLUMNS


def transpose(M):
    """Return the transposed matrix"""
    return [list(x) for x in zip(*M)]


def mean_square_error(U, V):
    """Return the mean square error of two vectors"""
    length = len(U)
    return sum(sub(U, V)[i] ** 2 for i in range(length)) / length


def argmax(V):
    """Return the index of the maximum value"""
    return V.index(max(V))


def categorical(label, classes=10):
    """Return a list of nine 0's and 1 where index equals the label"""
    x = [0] * classes
    x[label] = 1
    return x


def predict(network, image):
    """Return a vector of a network's predictions for each label"""
    A, b = network
    return add(multiply(image, A), b) 


def evaluate(network, images, labels):
    """Test the quality of a network and return performance statistics

    Arguments:
    network -- a network = (A, b) to evaluate
    images -- a list of images to test the network on
    labels -- a list of labels corresponding with the list 'images'

    Returns:
    predictions -- a list of all predicted labels
    totalcost -- the average cost of all predictions
    accuracy -- the percentage of correctly predicted images    
    """
    test_batch_size = 100
    test_batch = random.sample(list(zip(images, labels)), test_batch_size)
    images, labels = zip(*test_batch)

    number_of_images = len(images)
    predictions, correct, cost = [[0] * number_of_images for _ in range(3)]

    for idx, image in enumerate(images):
        prediction = predict(network, image_to_vector(image))
        predictions[idx] = argmax(prediction)
        if predictions[idx] == labels[idx]:
            correct[idx] = 1

        cost[idx] = mean_square_error(prediction, categorical(labels[idx]))

    accuracy = sum(correct) / number_of_images
    totalcost = sum(cost) / number_of_images
    return predictions, totalcost, accuracy


#### - PART 3 - ####
def create_batches(values, batch_size):
    """Shuffle the values, and return the shuffled values in batches

    Arguments:
    values -- a list with our images and labels
    batch_size -- The amount of batches that we want

    Returns: A list with images and labels that are shuffled and cut 
        into batches
    
    """
    random.shuffle(values)  
    return [values[i:i + batch_size] 
            for i in range(0, len(values), batch_size)]  


def update(network, images, labels):
    """Update the network with one step of gradient descend
    
    Arguments:
    network -- a network = (A, b) to update
    images -- a list of images in matrix form
    labels -- a list of correct labels corresponding to the images 
                in the list 'images'
    """
    x = [image_to_vector(image) for image in images]
    a = [predict(network, image) for image in x]
    y = [categorical(label) for label in labels]

    A, b = network
    step_size = 0.1
    n = len(images)
    scalar = step_size * (1 / (n*5))

    b[:] = sub(b, scalar_multiplication(scalar, [sum(a[img][j] - y[img][j]
                                                 for img in range(n))
                                                 for j in range(10)]))
    
    for i in range(len(A)):
        A[i] = sub(A[i], scalar_multiplication(scalar,[sum(x[img][i] * (a[img][j] - y[img][j])
                                                       for img in range(n))
                                                       for j in range(10)]))


def plot_network_updates(updates):
    """Plot the evolution of the accuracy and the cost 

    Arguments:
    updates -- a list of pairs (cost, accuracy) for each update

    Plot the accuracy and cost for each network evaluation, in two
        separate subplots
    """

    update_indices = list(range(len(updates)))
    
    cost, accuracy = zip(*updates)

    highest_acc = max(accuracy)

    plt.suptitle("Evolution of network")
    ax1 = plt.subplot(2, 1, 1)
    
    # Accuracy subplot
    ax1.plot(update_indices, accuracy)
    ax1.hlines(y=highest_acc, xmin=update_indices[0], xmax=update_indices[-1], 
               color="r", linestyle="dashed")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis= "x", labelbottom=False)
    ax1.text(update_indices[-1] // 2, highest_acc + 0.05, 
             f"{highest_acc}", color="r")
    
    # Cost subplot
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlabel("Network updates")
    ax2.set_ylabel("Cost")
    ax2.plot(update_indices, cost)

    plt.show()


def plot_weights(network):
    """Plot the network at the last iteration.

    Arguments:
    Network -- A network = (A, b)
    
    Plot 10 different plots for each label, 
        to see how the network identifies numbers
    """
    A = network[0]
    # Transposes the matrix, so every row is each label's
    #   corresponding pixel input weights
    label_weights = [[[label[row+column] for row in range(28)] 
                                         for column in range(0, 28*28, 28)] 
                                         for label in transpose(A)]
    
    plots = {} 
    for idx, image in enumerate(label_weights):
        plots[idx] = plt.subplot(2, 5, idx + 1)
        plots[idx].axes.get_yaxis().set_visible(False)
        plots[idx].axes.get_xaxis().set_visible(False)
        plt.imshow(image, cmap="plasma")
        plt.title(idx)

    plt.suptitle(
        "Linear classifier weights \n (columns of weight matrix A reshaped to image)"
        )
    plt.show()


def plot_images(images, labels, pred_labels=None):
    """Plot images with or without a network's predicted labels
    
    Arguments:
    images -- 10 images with 28 x 28 pixel values
    labels -- 10 integer label values for each of the images
    pred_labels -- Optional argument. 10 integer values, with the 
        predicted labels

    Plot 10 plots of the images and their corresponding correct labels.
        Optionally also show predicted labels.
    """
    if pred_labels is not None:
        # Check if length of all input vectors is 10, 
        #   then we only have 10 in the set.
        assert {len(images), len(labels), len(pred_labels)} == {10}, \
        "The function only takes 10 images, labels and predicted labels as argument"
    else:
        # Check if length of all input vectors is 10
        assert {len(images), len(labels)} == {10}, \
        "The function only takes 10 images, labels as argument"
    

    plots = {} # Store subplots to acess subplots' axises
    for idx, (image, label, pred_label) in enumerate(zip(images, labels, pred_labels)):
        plots[idx] = plt.subplot(2, 5, idx + 1)
        plots[idx].axes.get_yaxis().set_visible(False)
        plots[idx].axes.get_xaxis().set_visible(False)
        plt.imshow(image, cmap="binary")
        if pred_labels:
            plt.title(f"{label}" if label == pred_label else f"{pred_label}, correct {label}")
        else:
            plt.title(f"{label}")

    plt.suptitle("Images")
    plt.show()


def animate_weights(network_history):
    """Create a animation of how the weights changes for each iteration

    Argument:
    network_history -- A list of the network's weights = A
                        for each iteration

    Plot an animation, that showcases how the weights for each class 
        change after each update call
    """
    fig, subplots = plt.subplots(2,5)
    subplots = [plot for layer in subplots for plot in layer]

    # Same idea as in plot_weigths()
    all_image_weights = [[[[label[row+column] for row in range(28)]
                                              for column in range(0, 28*28, 28)]
                                              for label in transpose(A)]
                                              for A in network_history]
    
    image_weights = all_image_weights[0]

    def init_function():
        for idx, (subplot, image_weight) in enumerate(zip(subplots, image_weights)):
            subplot.axes.get_yaxis().set_visible(False)
            subplot.axes.get_xaxis().set_visible(False)
            subplot.set_title(f"{idx}")
            subplot.imshow(image_weight)

    def advance_plot(frame):
        image_weights = all_image_weights[frame]

        for idx, (subplot, image_weight) in enumerate(zip(subplots, image_weights)):
            subplot.clear()
            subplot.set_title(f"{idx}")
            subplot.imshow(image_weight)

    animation = FuncAnimation(fig,
                            func=advance_plot,
                            frames=range(0, len(network_history)),
                            interval=100,
                            repeat=False,
                            init_func=init_function
                            )
    
    plt.show()
    animation.save("weigths_animation.gif", fps=5)

def learn(images, labels, epochs, batch_size):
    """Train an initially random network on a set of training images
        and output various performance plots

    Arguments:
    images -- Takes the images we want to train on
    labels -- Takes the corresponding labels to the images
    epochs -- How many epochs we want to train over
    batch_size -- the batch size of the partitioned input
    
    Generate a random network = (A, b) and train the network with 
        repeated steps of gradient descent using the update function.
        
    Plot the best network, the evolution of the network, and the 
        predicted labels for the first 10 train images, aswell 
        as the animation for the network
    """
    init_b = [random.random() for _ in range(10)]
    init_A = [[random.random() / 784 for _ in range(10)] for _ in range(784)]

    results = [] # History for visualization
    network_history = [] # History for visualization

    network_history.append(init_A[:])

    network = (init_A, init_b)
    _, cost, accuracy = evaluate(network, testimages, testlabels)
    results.append((cost, accuracy))
    lowest_cost = cost
    
    for epoch in range(epochs):
        batches = create_batches(list(zip(images, labels)), batch_size)

        for batch_number, batch in enumerate(batches):
            batch_images, batch_labels = zip(*batch) # Unpack pairs into two lists

            print(f"Epoch: {epoch + 1}, Batch: {batch_number + 1}")
            update(network, batch_images, batch_labels)

            # The first few network updates experience the most change
            if batch_number + 1 < 10 and epoch == 0:
                network_history.append(network[0][:])

            if (batch_number + 1) % 10 == 0:
                _, cost, accuracy = evaluate(network, testimages, testlabels)
                results.append((cost, accuracy))

                if epoch == 0:
                    network_history.append(network[0][:])

                if cost < lowest_cost:
                    print("Better network found!")
                    linear_save("network.json", network)
                    lowest_cost = cost

                print(f"Accuracy: {accuracy * 100:.3} % \n")

    linear_save("network_history.json", network_history)

    best_network = linear_load("network.json")
    plot_weights(best_network)
    plot_network_updates(results)
    plot_images(testimages[:10], testlabels[:10], 
                [argmax(predict(best_network, 
                image_to_vector(image))) for image in testimages[:10]])

# animate_weights(linear_load("network_history.json"))

trainimages = read_images("train-images-idx3-ubyte.gz")
trainlabels = read_labels("train-labels-idx1-ubyte.gz")
testimages = read_images("t10k-images-idx3-ubyte.gz")
testlabels = read_labels("t10k-labels-idx1-ubyte.gz")
learn(trainimages, trainlabels, 1, 100)
