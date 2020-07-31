from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Flatten, Dense, Activation, Conv2D, MaxPooling2D, Concatenate, concatenate
import flexflow.keras.optimizers
from flexflow.keras.datasets import mnist
from flexflow.keras.datasets import cifar10
from flexflow.keras import losses
from flexflow.keras import metrics

import flexflow.core as ff
import numpy as np
import argparse
import gc
  
def top_level_task():
  num_classes = 10

  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  x_train = x_train.reshape(60000, 784)
  x_train = x_train.astype('float32')
  x_train /= 255
  y_train = y_train.astype('int32')
  y_train = np.reshape(y_train, (len(y_train), 1))
  print("shape: ", x_train.shape)
  
  #teacher
  
  input_tensor1 = Input(shape=(784,), dtype="float32")
  
  d1 = Dense(512, input_shape=(784,), activation="relu")
  d2 = Dense(512, activation="relu")
  d3 = Dense(num_classes)
  
  output = d1(input_tensor1)
  output = d2(output)
  output = d3(output)
  output = Activation("softmax")(output)
  
  teacher_model = Model(input_tensor1, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  teacher_model.compile(optimizer=opt)

  teacher_model.fit(x_train, y_train, epochs=1)
  
  d1_kernel, d1_bias = d1.get_weights(teacher_model.ffmodel)
  d2_kernel, d2_bias = d2.get_weights(teacher_model.ffmodel)
  d3_kernel, d3_bias = d3.get_weights(teacher_model.ffmodel)
  
  # student
  
  input_tensor2 = Input(shape=(784,), dtype="float32")
  
  sd1_1 = Dense(512, input_shape=(784,), activation="relu")
  sd2 = Dense(512, activation="relu")
  sd3 = Dense(num_classes)
  
  output = sd1_1(input_tensor2)
  output = sd2(output)
  output = sd3(output)
  output = Activation("softmax")(output)

  student_model = Model(input_tensor2, output)

  opt = flexflow.keras.optimizers.SGD(learning_rate=0.01)
  student_model.compile(optimizer=opt)
  
  sd1_1.set_weights(student_model.ffmodel, d1_kernel, d1_bias)
  sd2.set_weights(student_model.ffmodel, d2_kernel, d2_bias)
  sd3.set_weights(student_model.ffmodel, d3_kernel, d3_bias)

  student_model.fit(x_train, y_train, epochs=1)

if __name__ == "__main__":
  print("Functional API, mnist mlp teach student")
  top_level_task()
  gc.collect()