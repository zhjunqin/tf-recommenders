# -*- coding:utf-8 -*-
from enum import Enum
import tensorflow as tf



class Config(object):


    mmoe_networks = {
      'user_input_layer': [
        'user_id_embedding',
        'cms_segid_embedding',
        'cms_group_id_indicator',
        'final_gender_code_indicator',
        'age_level_indicator',
        'pvalue_level_indicator',
        'shopping_level_indicator',
        'occupation_indicator',
        'new_user_class_level_indicator'],
      'item_input_layer': [
        'adgroup_id_embedding',
        'cate_id_embedding',
        'customer_id_embedding',
        'campaign_id_embedding',
        'brand_id_embedding',
        'pid_indicator'],
      'input_layer': ['user_input_layer', 'item_input_layer'],
      'input_hidden_layers': [],
      'experts_layers': [512],
      'num_experts': 5,
      'towers': [
        {'name': 'tower_0', 'hidden_layers' : [256, 128], 'output_weight': 1.0,'target': 'clk'},
        {'name': 'tower_1', 'hidden_layers' : [256, 128], 'output_weight': 1.0, 'target': 'price'},
        ]
    }

    dataset_setting = {
      'batch_size': 1024,
      'train_epochs': 2,
      'cycle_length': 8,
      'shuffle_buffer': 10000,
      'num_parallel_calls': 8
    }

    feature_columns = {
        'user_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'user_id',
                'hash_bucket_size': 1061768,
                'dtype': tf.int64,
            }
        },
        'cms_segid': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'cms_segid',
                'hash_bucket_size': 97,
                'dtype': tf.int64,
            }
        },
        'cms_group_id': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
                'key': 'cms_group_id',
                'num_buckets': 13,
                'default_value': 0,
            }
        },
        'final_gender_code': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
                'key': 'final_gender_code',
                'num_buckets': 2,
                'default_value': 0,
            }
        },
        'age_level': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
                'key': 'age_level',
                'num_buckets': 7,
                'default_value': 0,
            }
        },
        'pvalue_level': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
                'key': 'pvalue_level',
                'num_buckets': 4,
                'default_value': 0,
            }
        },
        'shopping_level': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
                'key': 'shopping_level',
                'num_buckets': 3,
                'default_value': 0,
            }
        },
        'occupation': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
              'key': 'occupation',
              'num_buckets': 2,
              'default_value': 0,
            }
        },
        'new_user_class_level': {
            'type': 'categorical_column_with_identity',
            'parameters' : {
              'key': 'new_user_class_level',
              'num_buckets': 5,
              'default_value': 0,
            }
        },
        'adgroup_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
              'key': 'adgroup_id',
              'hash_bucket_size': 827009,
              'dtype': tf.int64,
            }
        },
        'cate_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'cate_id',
                'hash_bucket_size': 6725,
            'dtype': tf.int64,
            }
        },
        'customer_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'customer_id',
                'hash_bucket_size': 252841,
            'dtype': tf.int64,
            }
        },
        'campaign_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'campaign_id',
                'hash_bucket_size': 417656,
            'dtype': tf.int64,
            }
        },
        'brand_id': {
            'type': 'categorical_column_with_hash_bucket',
            'parameters' : {
                'key': 'brand_id',
                'hash_bucket_size': 98772,
            'dtype': tf.int64,
            }
        },
        'pid': {
            'type': 'categorical_column_with_vocabulary_list',
            'parameters' : {
                'key': 'pid',
                'vocabulary_list': ['430548_1007', '430539_1007'],
            'dtype': tf.string,
            }
        },

        # Embedding
        'user_id_embedding': {
            'type': 'embedding_column',
            'depends': ['user_id'],
            'parameters' : {
            }
        },
        'cms_segid_embedding': {
            'type': 'embedding_column',
            'depends': ['cms_segid'],
            'parameters' : {
            }
        },
        'cms_group_id_indicator': {
            'type': 'indicator_column',
            'depends': ['cms_group_id'],
            'parameters' : {
            }
        },
        'final_gender_code_indicator': {
            'type': 'indicator_column',
            'depends': ['final_gender_code'],
            'parameters' : {
            }
        },
        'age_level_indicator': {
            'type': 'indicator_column',
            'depends': ['age_level'],
            'parameters' : {
            }
        },
        'pvalue_level_indicator': {
            'type': 'indicator_column',
            'depends': ['pvalue_level'],
            'parameters' : {
            }
        },
        'shopping_level_indicator': {
            'type': 'indicator_column',
            'depends': ['shopping_level'],
            'parameters' : {
            }
        },
        'occupation_indicator': {
            'type': 'indicator_column',
            'depends': ['occupation'],
            'parameters' : {
            }
        },
        'new_user_class_level_indicator': {
            'type': 'indicator_column',
            'depends': ['new_user_class_level'],
            'parameters' : {
            }
        },

        'adgroup_id_embedding': {
            'type': 'embedding_column',
            'depends': ['adgroup_id'],
            'parameters' : {
            }
        },
        'cate_id_embedding': {
            'type': 'embedding_column',
            'depends': ['cate_id'],
            'parameters' : {
            }
        },
        'customer_id_embedding': {
            'type': 'embedding_column',
            'depends': ['customer_id'],
            'parameters' : {
            }
        },
        'campaign_id_embedding': {
            'type': 'embedding_column',
            'depends': ['campaign_id'],
            'parameters' : {
            }
        },
        'brand_id_embedding': {
            'type': 'embedding_column',
            'depends': ['brand_id'],
            'parameters' : {
            }
        },
        'pid_indicator': {
            'type': 'indicator_column',
            'depends': ['pid'],
            'parameters' : {
            }
        },
    }

    label_features = {
        'clk': tf.FixedLenFeature(shape=(), dtype=tf.int64),
        'price': tf.FixedLenFeature(shape=(), dtype=tf.float32),
    }

    input_features = {
        # User fix len feature
        'user_id': tf.FixedLenFeature([], tf.int64),
        'cms_segid': tf.FixedLenFeature((), tf.int64, 0),
        'cms_group_id': tf.FixedLenFeature((), tf.int64, 0),
        'final_gender_code': tf.FixedLenFeature((), tf.int64, 0),
        'age_level': tf.FixedLenFeature((), tf.int64, 0),
        'pvalue_level': tf.FixedLenFeature((), tf.int64, 0),
        'shopping_level': tf.FixedLenFeature((), tf.int64, 0),
        'occupation': tf.FixedLenFeature((), tf.int64, 0),
        'new_user_class_level': tf.FixedLenFeature((), tf.int64, 0),

        # User var len feature
        # pass

        # Item fix len feature
        'adgroup_id': tf.FixedLenFeature((), tf.int64, 0),
        'cate_id': tf.FixedLenFeature((), tf.int64, 0),
        'campaign_id': tf.FixedLenFeature((), tf.int64, 0),
        'customer_id': tf.FixedLenFeature((), tf.int64, 0),
        'pid': tf.FixedLenFeature((), tf.string),

        # Item var len feature
        'brand_id': tf.VarLenFeature(tf.int64),
    }

    all_features = {**label_features, **input_features}
