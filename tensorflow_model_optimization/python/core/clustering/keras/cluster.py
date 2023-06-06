from typing import Union, List

def _add_clustering_wrapper(layer: Union[Layer, tf.keras.Model]) -> Union[Layer, tf.keras.Model]:
    """Modifies a keras layer or model to be clustered during training."""
    if isinstance(layer, tf.keras.Model):
        # Check whether the model is a subclass.
        if _is_not_supported_model(layer):
            raise ValueError('Subclassed models are not supported currently.')
        return _clone_model(layer, _add_clustering_wrapper)
    
    if isinstance(layer, cluster_wrapper.ClusterWeights):
        return layer
    if isinstance(layer, InputLayer):
        return _get_input_layer(layer)
    if _is_rnn_or_bidirectional(layer):
        return _get_cluster_weights_rnn(layer)
    if isinstance(layer, tf.keras.layers.MultiHeadAttention):
        return _get_cluster_weights_mha(layer)

    # Skip clustering if Conv2D layer has insufficient number of weights
    # for type of clustering
    if _should_skip_clustering(layer):
        return layer

    return _get_cluster_weights(layer)

# ...

def _cluster_weights(to_cluster: Union[Layer, tf.keras.Model, List[Layer]],
                     number_of_clusters: int,
                     cluster_centroids_init: CentroidInitialization,
                     preserve_sparsity: bool = False,
                     cluster_per_channel: bool = False,
                     **kwargs) -> Union[Layer, tf.keras.Model, List[Layer]]:
    """Modifies a keras layer or model to be clustered during training."""
    if not clustering_centroids.CentroidsInitializerFactory.init_is_supported(
        cluster_centroids_init):
        raise ValueError('Cluster centroid initialization {} not supported'.format(cluster_centroids_init))

    if isinstance(to_cluster, tf.keras.Model):
        return _clone_model(to_cluster, _add_clustering_wrapper)
    if isinstance(to_cluster, Layer):
        return _add_clustering_wrapper(to_cluster)
    if isinstance(to_cluster, list):
        return _wrap_list(to_cluster, _add_clustering_wrapper)
    
# ...
