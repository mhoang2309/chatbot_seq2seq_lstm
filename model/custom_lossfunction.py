import tensorflow as tf

def custom_lossfunction(real, pred):
    # Custom loss function that will not consider the loss for padded zeros.
    # Refer https://www.tensorflow.org/tutorials/text/nmt_with_attention#define_the_optimizer_and_the_loss_function
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
 
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
 
    return tf.reduce_mean(loss_)