import numpy as np
import tensorflow as tf
    
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

    # -----------------------------------------------------------------------------
    
    # -------------------- Step3. Create Tensorflow module(graphic) ---------------

    model = tf.keras.models.Sequential( [ tf.keras.layers.Flatten(),
                                          tf.keras.layers.Dense(256, activation=tf.nn.relu),
                                          tf.keras.layers.Dense(10),
                                          tf.keras.layers.Softmax() ] ) # define layer

    #simple print model
    temp_inputs = tf.keras.Input(shape=(28, 28, 1))
    model(temp_inputs)
    model.summary()
    
    # loss function
    
    # learning decay
    lr_decay = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate,decay_steps=x_train.shape[0] / batch_size * 5 * 5,decay_rate=0.5,staircase=True)

    # Optimizer and loss function
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_decay),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    # -----------------------------------------------------------------------------
    
    # -------------------- Step4. Run Graphic -------------------------------------
    model.fit(x_train, y_train, epochs=10)

    model.evaluate(x_test, y_test)    
    
    # -----------------------------------------------------------------------------
            
    print( "Finished mnist learning" )

# End main()

if __name__ == "__main__":
    main()


