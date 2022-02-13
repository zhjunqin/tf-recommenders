import tensorflow as tf


class DNN(tf.keras.Model):
    """ 深度模型

    """
    def __init__(self, feature_columns, layer_sizes):

        super().__init__()
        self.feature_columns = feature_columns
        self.input_features = tf.keras.layers.DenseFeatures(self.feature_columns)
        self.dense_layers = tf.keras.Sequential()
        self.dense_layers.add(tf.keras.layers.Dense(layer_sizes[0], activation='relu'))
        for layer_size in layer_sizes[1:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation='relu'))
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.input_features(inputs)
        return self.dense_layers(x)



class DSSM(tf.keras.Model):
    """
    双塔模型
    """
    def __init__(self, item_features, user_features, user_layer_sizes, item_layer_sizes, num_hard_negatives=None):
        super().__init__()
        self.query_model = DNN(user_features, user_layer_sizes)
        self.candidate_model = DNN(item_features, item_layer_sizes)
        self._loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.SUM)

    def train_step(self, inputs):
        """Custom train step using the `compute_loss` method."""

        with tf.GradientTape() as tape:
            loss = self.compute_loss(inputs, training=True)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, inputs):
        """Custom test step using the `compute_loss` method."""

        loss = self.compute_loss(inputs, training=False)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics


    @tf.function
    def compute_loss(self, inputs, training: bool = False) -> tf.Tensor:
        """
        计算loss
        :param inputs:
        :param training:
        :return:
        """
        query_embeddings = self.query_model(inputs)
        query_embeddings = self.candidate_model(inputs)

        scores = tf.linalg.matmul(
            query_embeddings, query_embeddings, transpose_b=True)
        num_queries = tf.shape(scores)[0]
        num_candidates = tf.shape(scores)[1]
        generate_labels = tf.eye(num_queries, num_candidates)
        loss = self._loss(y_pred=scores, y_true=generate_labels)

        return loss

