dataset：
python download_and_convert_data.py \ --dataset_name=flowers \ --dataset_dir=./tmp/data/flowers
 
train:
python train_image_classifier.py  \ --clone_on_cpu=True \ --train_dir=./tmp/train_logs1 \ --dataset_name=flowers \ --dataset_split_name=train \ --dataset_dir=./tmp/data/flowers \ --model_name=inception_v3 \ --max_number_of_steps=100

eval:
python eval_image_classifier.py \ --alsologtostderr \ --checkpoint_path=./tmp/train_logs \ --dataset_dir=./tmp/data/flowers \ --dataset_name=flowers \ --dataset_split_name=validation \ --model_name=inception_v3
 
save model:
python export_inference_graph.py \ --alsologtostderr \ --model_name=inception_v3 \ --output_file=./tmp/inception_v3_inf_graph.pb \ --dataset_name=flowers

python freeze_graph.py \ --input_graph ./tmp/inception_v3_inf_graph.pb \ --input_checkpoint .tmp/train_logs/model.ckpt-100 \ --input_binary true \ --output_node_names InceptionV3/Predictions/Reshape_1 \ --output_graph .tmp/frozen_graph.pb
