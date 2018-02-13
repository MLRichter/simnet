import numpy as np
import os, struct

class MNIST():
    def __init__(self, directory):
        self._directory = directory
        
        self._training_data = self._load_binaries("train-images.idx3-ubyte")
        self._training_labels = self._load_binaries("train-labels.idx1-ubyte")
        self._test_data = self._load_binaries("t10k-images.idx3-ubyte")
        self._test_labels = self._load_binaries("t10k-labels.idx1-ubyte")
        
        np.random.seed(0)
        samples_n = self._training_labels.shape[0]
        random_indices = np.random.choice(samples_n, samples_n // 10, replace = False)
        np.random.seed()
        
        self._validation_data = self._training_data[random_indices]
        self._validation_labels = self._training_labels[random_indices]
        self._training_data = np.delete(self._training_data, random_indices, axis = 0)
        self._training_labels = np.delete(self._training_labels, random_indices)

        self._create_class_map()
    
    def _load_binaries(self, file_name):
        path = os.path.join(self._directory, file_name)
        
        with open(path, 'rb') as fd:
            check, items_n = struct.unpack(">ii", fd.read(8))

            if "images" in file_name and check == 2051:
                height, width = struct.unpack(">II", fd.read(8))
                images = np.fromfile(fd, dtype = 'uint8')
                return np.reshape(images, (items_n, height, width))
            elif "labels" in file_name and check == 2049:
                return np.fromfile(fd, dtype = 'uint8')
            else:
                raise ValueError("Not a MNIST file: " + path)
    
    
    def get_training_batch(self, batch_size):
        return self._get_batch(self._training_data, self._training_labels, batch_size)
    
    def get_validation_batch(self, batch_size):
        return self._get_batch(self._validation_data, self._validation_labels, batch_size)
    
    def get_test_batch(self, batch_size):
        return self._get_batch(self._test_data, self._test_labels, batch_size)
    
    def _get_batch(self, data, labels, batch_size):
        samples_n = labels.shape[0]
        if batch_size <= 0:
            batch_size = samples_n
        
        random_indices = np.random.choice(samples_n, samples_n, replace = False)
        data = data[random_indices]
        labels = labels[random_indices]
        for i in range(samples_n // batch_size):
            on = i * batch_size
            off = on + batch_size
            yield data[on:off], labels[on:off]
    
    
    def get_sizes(self):
        training_samples_n = self._training_labels.shape[0]
        validation_samples_n = self._validation_labels.shape[0]
        test_samples_n = self._test_labels.shape[0]
        return training_samples_n, validation_samples_n, test_samples_n

    def get_samples(self,num_samples):
        """
        get a number of random samples from the training set
        :param num_samples:
        :return:
        """
        random_indices = np.random.choice(len(self._training_labels), num_samples, replace = False)
        return self._training_data[random_indices], self._training_labels[random_indices]

    def _create_class_map(self):
        """
        create a list with 10 entries, belonging to the 10 classes. Each entry contains a list of all
        data points with the corresponding class. This function is applied on the training-set only
        :return:
        """
        class_map = []
        for i in range(10):
            class_map.append(self._training_data[self._training_labels == i])
        self.class_map = class_map

    def get_classification_samples(self, n_samples_per_class):
        """
        return a data-vector with n samples per class, used for one shot classification
        :param n_samples_per_class:
        :return:
        """
        data = []
        labels = []

        for i,class_samples in enumerate(self.class_map):
            random_indices = np.random.choice(len(class_samples),n_samples_per_class,replace=False)
            data.append(class_samples[random_indices])
            labels.append(np.ones(n_samples_per_class)*i)
        return np.asarray(data), np.asarray(labels)





