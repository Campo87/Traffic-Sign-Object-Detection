
import os

my_model = "my_efficientdet_d0"
training_script = os.path.join("Tensorflow/models/research/object_detection/model_main_tf2.py")

os.system(f"python {training_script} --model_dir={my_model}/checkpoint --pipeline_config_path={my_model}/pipeline.config --checkpoint_dir={my_model}/checkpoint")
