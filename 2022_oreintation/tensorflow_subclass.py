import numpy as np
import tensorflow as tf

"""
reference: https://www.tensorflow.org/tutorials/quickstart/advanced?hl=en
customize layer: https://www.tensorflow.org/tutorials/customization/custom_layers?hl=en
start guide: https://www.tensorflow.org/guide/basic_training_loops
"""
class mn_Model(tf.keras.Model): #Inherit tf.keras.Model class
    def __init__(self):
      super(mn_Model, self).__init__()    
      # init. your layers or inherit layer.   
      self.flat = tf.keras.layers.Flatten()
      self.dense1 = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)
      
    # end def  
     
    def call(self, inputs, training=False):
      #define your forwarding network
      net = self.flat(inputs)
      net = self.dense1(net)
      net = self.dense2(net)
      return net
      
    # end def

# end class

"""
training four step:

1.send input to model and get output
2.use output to calculate loss
3.calculate gradient by loss
4.use gradient optimize your model
"""

# Loss function
def loss_func( logits, images, labels ):
    all_loss = tf.keras.losses.sparse_categorical_crossentropy( y_true=tf.argmax(labels, 1), y_pred=logits )
    loss = tf.reduce_mean( all_loss )
    return loss
    
# end loss_func

def training_model( model, images, labels, optimizer ):
    with tf.GradientTape() as tape:
        output = model(images, training=True) # Sending a batch of inputs through the model to generate outputs
        loss = loss_func( output, images, labels ) # Calculating the loss by comparing the outputs to the output (or label)
    # end with
    
    gradients = tape.gradient( loss, model.trainable_variables ) # Using gradient tape to find the gradients
    optimizer.apply_gradients( zip( gradients, model.trainable_variables ) ) # Optimizing the variables with those gradients
    # optimizer.apply_gradients( zip( gradients, model.variables ) ) I don't know where is different
    
    return loss
    
# end training_model()

# Evaluation Function
def evaluate(model, images, labels):    
    predicts = model(images, training=False)
    correct_predict = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
    return accuracy
    
# end evaluate
    
def main():

    # -------------------- Define Param -------------------------------------------
    learning_rate = 0.001
    training_epochs = 30
    batch_size = 300
    # -----------------------------------------------------------------------------
    # -------------------- Step1. Read Data, Read Data from CSV -------------------
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    TotalDataCount = len( x_train )+len( x_test )
    TraningDataCount = len( x_train )
    TestDataCount = len( x_test )
    # ----------------------------------------------------------------------------
    
    # -------------------- Step2. Split Input/Output Data  ------------------------
    x_train = x_train/255
    x_test = x_test/255
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    # Build dataset pipeline
    # cut batch
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=100000).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    # -----------------------------------------------------------------------------
    
    # -------------------- Step3. Create Tensorflow module(graphic) ---------------

    model = mn_Model() # define layer

    #simple print model
    temp_inputs = tf.keras.Input(shape=(28, 28, 1))
    model(temp_inputs)
    model.summary()
    
    # learning decay
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,decay_steps=x_train.shape[0] / batch_size * 5 * 5,decay_rate=0.5,staircase=True)

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_decay)
    # -----------------------------------------------------------------------------
    
    # -------------------- Step4. Run Graphic -------------------------------------
    for epoch in range(training_epochs):
        avg_loss = 0.
        avg_train_acc = 0.
        avg_test_acc = 0.
        train_step = 0
        test_step = 0
    
        for images, labels in train_ds: # training            
            loss = training_model( model, images, labels, optimizer ) # training and update loss
            
            acc = evaluate(model, images, labels) # calculate accuracy every batch            
            avg_loss = avg_loss + loss
            avg_train_acc = avg_train_acc + acc
            train_step += 1
        # end for
            
        avg_loss = avg_loss / train_step
        avg_train_acc = avg_train_acc / train_step
    
        for images, labels in test_ds: # testing
            acc = evaluate(model, images, labels) # calculate accuracy every batch
            avg_test_acc = avg_test_acc + acc
            test_step += 1    
        # end for
            
        avg_test_acc = avg_test_acc / test_step    

        print('Epoch:', '{}'.format(epoch + 1), 
              'loss =', '{:.8f}'.format(avg_loss), 
              'train accuracy = ', '{:.4f}'.format(avg_train_acc), 
              'test accuracy = ', '{:.4f}'.format(avg_test_acc))    
    
    # end for
    # -----------------------------------------------------------------------------
            
    print( "Finished mnist learning" )

# End main()

if __name__ == "__main__":
    main()


