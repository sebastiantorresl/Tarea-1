import mnist_loader
import networka
import pickle
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net = networka.Network([784, 20, 10], cost=networka.CrossEntropyCost)
net.large_weight_initializer()
net.SGD(training_data, 30, 20, 0.25, evaluation_data=test_data, monitor_evaluation_accuracy=True)
archivo = open("red_prueba.pkl", 'wb')
pickle.dump(net, archivo)
archivo.close()
exit()
# leer el archivo
archivo_lectura = open("red_prueba.pkl", 'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
net.SGD(training_data, 30, 20, 0.25, evaluation_data=test_data)
archivo = open("red_prueba.pkl", 'wb')
pickle.dump(net, archivo)
archivo.close()
exit()
