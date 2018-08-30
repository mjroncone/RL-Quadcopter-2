from keras import layers, models, optimizers
from keras import backend

# Actor model based on https://classroom.udacity.com/nanodegrees/nd009t/parts/ac12e0fe-e54e-40d5-b0f8-136dbdd1987b/modules/691b7845-f7d8-413d-90c7-971cd5016b5c/lessons/fef7e79a-0941-460b-936c-d24c759ff700/concepts/d254347a-68f4-47d0-912a-33fd79719cf8#

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        """
        Params
        ======
            state_size (int): Number of dimensions for each state
            action_size (int): Number of dimensions for each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """

        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low


        self.build_model()

    def build_model(self):
        """Returns a policy network which maps states to actions"""
        kstates = layers.Input(shape = (self.state_size, ), name = 'states')

        net = layers.Dense(units=32, activation='relu')(states)
        net = layers.Dense(units=64, activation='relu')(net)
        net = layers.Dense(units=32, activation='relu')(net)

        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
                name='raw_actions')(net)

        # Scales the [0, 1] output for each action dimension to the proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
                name='actions')(raw_actions)

        self.model = models.Model(inputs=staes, outputs=actions)

        # Defines the loss function using action value gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = backend.mean(-action_gradients * actions)

        optimizer = optimizers.Adam()
        updates_operation = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_function = backend.function(
                inputs=[self.model.input, action_gradients, backend.learning_phase()],
                outputs=[],
                updates=updates_operation)
