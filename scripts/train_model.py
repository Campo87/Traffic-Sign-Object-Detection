import os

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util

my_model = "my_models/my_ssd_resnet152_v1"
pre_trained_model = "pre-trained/ssd_resnet152_v1"

config = config_util.get_configs_from_pipeline_file(os.path.join(my_model, "pipeline.config"))

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(os.path.join(my_model, "pipeline.config"), "r") as f:                                                                                                                                                                                                                     
    proto_str = f.read()                                                                                                                                                                                                                                          
    text_format.Merge(proto_str, pipeline_config)  

pipeline_config.model.ssd.num_classes = 1
pipeline_config.train_config.batch_size = 2
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(pre_trained_model, 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path= "label_map.pbtxt"
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = ["images/train.record"]
pipeline_config.eval_input_reader[0].label_map_path = "label_map.pbtxt"
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = ["images/test.record"]

config_text = text_format.MessageToString(pipeline_config)                                                                                                                                                                                                        
with tf.io.gfile.GFile(f"{my_model}/pipeline.config", "wb") as f:                                                                                                                                                                                                                     
    f.write(config_text)   

training_script = os.path.join("Tensorflow/models/research/object_detection/model_main_tf2.py")

os.system(f"python {training_script} --model_dir={my_model} --pipeline_config_path={my_model}/pipeline.config --num_train_steps=2000")
